import * as THREE from "three";
import type { Model3DPart, PartCheck } from "../../types";

/*
 * Plays procedures authored in Vishwa (exported .procedure.json, schema v2)
 * on a loaded assembly: each step's part animations, and which camera facts
 * (part present / absent) mark the step done.
 *
 * Only the subset TVASTA can show over a tracked object is played: translate,
 * rotate (axis + angle), highlight and pulse. Camera moves, fades, isolate,
 * etc. are Vishwa-viewer concerns and are listed as skipped. Semantics follow
 * Vishwa's TrackEvaluator: every action starts from the pose its part has
 * when the action begins, `hold` keeps the end pose afterwards (otherwise the
 * part returns), `reverse` plays the path end -> start.
 */

export type Vec3 = [number, number, number];

export interface ProcedureAction {
  id: string;
  type: string;
  duration?: number;
  delay?: number;
  targetParts?: string[];
  easing?: string;
  reverse?: boolean;
  hold?: boolean;
  by?: Vec3;
  from?: Vec3;
  to?: Vec3;
  axis?: Vec3;
  angleDeg?: number;
  color?: string;
  pulseIntensity?: number;
}

export interface ProcedureStep {
  id: string;
  title?: string;
  description?: string;
  duration?: number;
  actions?: ProcedureAction[];
  expectedState?: { modelId: string; stateId: string }[];
}

export interface Procedure {
  id: string;
  title?: string;
  description?: string;
  stateModels?: {
    id: string;
    states?: { id: string; label?: string; allOf?: { kind: string; part?: string; source?: string }[] }[];
  }[];
  steps: ProcedureStep[];
}

export interface ProcedurePackage {
  schemaVersion: string;
  procedure: Procedure;
}

export const PLAYED_ACTIONS = new Set(["translate", "rotate", "highlight", "pulse"]);

const EASINGS: Record<string, (t: number) => number> = {
  linear: (t) => t,
  easeInQuad: (t) => t * t,
  easeOutQuad: (t) => t * (2 - t),
  easeInOutQuad: (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2),
  easeInCubic: (t) => t ** 3,
  easeOutCubic: (t) => 1 - (1 - t) ** 3,
  easeInOutCubic: (t) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2),
  easeOutBack: (t) => {
    const c1 = 1.70158;
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2;
  },
  stepStart: (t) => (t > 0 ? 1 : 0),
  stepEnd: (t) => (t >= 1 ? 1 : 0),
};

const actionEnd = (a: ProcedureAction) => (a.delay ?? 0) + (a.duration ?? 0);

/** How long a step's animation runs, ms: its declared duration or its last action's end. */
export function stepDuration(step: ProcedureStep): number {
  return Math.max(step.duration ?? 0, ...(step.actions ?? []).map(actionEnd), 0);
}

/** Action types in the procedure that TVASTA doesn't play (e.g. cameraFocus). */
export function skippedActionTypes(procedure: Procedure): string[] {
  const types = new Set<string>();
  for (const step of procedure.steps) for (const a of step.actions ?? []) if (!PLAYED_ACTIONS.has(a.type)) types.add(a.type);
  return [...types];
}

export interface VisionFact {
  kind: "partPresent" | "partAbsent";
  part: string;
}

/** The camera-checkable facts of a step's expected end state (mirrors the backend's expected_vision_facts). */
export function expectedVisionFacts(procedure: Procedure, step: ProcedureStep): VisionFact[] {
  const facts: VisionFact[] = [];
  for (const ref of step.expectedState ?? []) {
    const state = procedure.stateModels?.find((m) => m.id === ref.modelId)?.states?.find((s) => s.id === ref.stateId);
    for (const f of state?.allOf ?? []) {
      if ((f.kind === "partPresent" || f.kind === "partAbsent") && f.part) facts.push({ kind: f.kind, part: f.part });
    }
  }
  return facts;
}

export type FactStatus = "met" | "not_met" | "uncertain" | "unknown";

/** One fact against the latest camera part checks (by part name, as calibrated in the Parts panel). */
export function factStatus(fact: VisionFact, checks: PartCheck[] | null): FactStatus {
  const check = checks?.find((c) => c.part_name === fact.part);
  if (!check) return "unknown";
  if (check.state === "uncertain") return "uncertain";
  return (check.state === "absent") === (fact.kind === "partAbsent") ? "met" : "not_met";
}

/** Progress of one step's camera check, frame by frame (see stepCheck). */
export interface StepCheck {
  armed: boolean; // starting state confirmed: the end state was seen NOT reached yet
  startFrames: number; // consecutive frames showing the starting state
  endFrames: number; // consecutive frames showing the end state, once armed
}

export const NEW_STEP_CHECK: StepCheck = { armed: false, startFrames: 0, endFrames: 0 };

/**
 * One camera frame's fact statuses applied to a step's check. A step passes
 * only after `required` consecutive frames of its starting state (every fact
 * clearly read, at least one not true yet — e.g. the cap still on) and THEN
 * `required` consecutive frames of its end state. Without the first part a
 * step whose end state already holds (refitting a cap that never came off),
 * or a burst of misreadings, would pass without anything happening.
 * Unclear frames (uncertain / not tracking) break either run.
 */
export function stepCheck(check: StepCheck, statuses: FactStatus[], required: number): { check: StepCheck; done: boolean } {
  if (!check.armed) {
    const startState = statuses.includes("not_met") && statuses.every((s) => s === "met" || s === "not_met");
    const startFrames = startState ? check.startFrames + 1 : 0;
    return { check: { armed: startFrames >= required, startFrames, endFrames: 0 }, done: false };
  }
  const endFrames = statuses.length > 0 && statuses.every((s) => s === "met") ? check.endFrames + 1 : 0;
  return { check: { ...check, endFrames }, done: endFrames >= required };
}

/**
 * Procedure part name -> the loaded object: the name given in the Parts
 * panel, else the GLB node name (as authored or as GLTFLoader sanitized it).
 */
export function makePartResolver(parts: Model3DPart[], objects: Map<number, THREE.Object3D>) {
  return (name: string): THREE.Object3D | null => {
    const part = parts.find((p) => p.display_name === name) ?? parts.find((p) => p.node_name === name);
    if (part) return objects.get(part.node_index) ?? null;
    for (const obj of objects.values()) if (obj.name === name || obj.name === name.replace(/[^\w-]/g, "")) return obj;
    return null;
  };
}

const PROCEDURE_OUTLINE = "tvastaProcedureOutline";

export class ProcedurePlayer {
  private rest = new Map<THREE.Object3D, { position: THREE.Vector3; quaternion: THREE.Quaternion }>();
  private outlines = new Map<THREE.Object3D, THREE.LineSegments[]>();
  readonly unmatched: string[] = [];

  constructor(
    private readonly procedure: Procedure,
    private readonly resolve: (name: string) => THREE.Object3D | null
  ) {
    for (const step of procedure.steps) {
      for (const action of step.actions ?? []) {
        for (const name of action.targetParts ?? []) {
          const obj = resolve(name);
          if (!obj) {
            if (!this.unmatched.includes(name)) this.unmatched.push(name);
          } else if (!this.rest.has(obj)) {
            this.rest.set(obj, { position: obj.position.clone(), quaternion: obj.quaternion.clone() });
          }
        }
      }
    }
  }

  /** Poses the parts as they are `timeMs` into step `index`, with all earlier steps completed. */
  seek(index: number, timeMs: number): void {
    this.reset(false);
    this.procedure.steps.forEach((step, i) => {
      if (i > index) return;
      const time = i < index ? Infinity : timeMs;
      // In start order, so a later action on the same part starts from where
      // the earlier one left it (Vishwa snapshots each action at its start).
      const actions = [...(step.actions ?? [])].sort((a, b) => (a.delay ?? 0) - (b.delay ?? 0));
      for (const action of actions) this.apply(action, time, i === index);
    });
  }

  /** Puts every animated part back where it was when the procedure loaded. */
  reset(removeOutlines = true): void {
    this.rest.forEach((pose, obj) => {
      obj.position.copy(pose.position);
      obj.quaternion.copy(pose.quaternion);
    });
    this.outlines.forEach((lines) =>
      lines.forEach((line) => {
        line.visible = false;
        if (removeOutlines) {
          line.removeFromParent();
          line.geometry.dispose();
          (line.material as THREE.Material).dispose();
        }
      })
    );
    if (removeOutlines) this.outlines.clear();
  }

  private progress(action: ProcedureAction, time: number): number | null {
    const start = action.delay ?? 0;
    const duration = action.duration ?? 0;
    let linear: number;
    if (time < start) linear = 0;
    else if (time >= start + duration) {
      if (!action.hold) return null; // finished, not held: part is back at its start pose
      linear = 1;
    } else linear = duration > 0 ? (time - start) / duration : 1;
    const eased = (EASINGS[action.easing ?? "linear"] ?? EASINGS.linear)(linear);
    return action.reverse ? 1 - eased : eased;
  }

  private apply(action: ProcedureAction, time: number, current: boolean): void {
    const targets = (action.targetParts ?? []).map(this.resolve).filter((o): o is THREE.Object3D => o !== null);
    if (action.type === "translate" || action.type === "rotate") {
      // Before it starts a non-reversed action is at rest; a reversed one already sits at its far end.
      if (time < (action.delay ?? 0) && !action.reverse) return;
      const p = this.progress(action, time);
      if (p === null) return;
      for (const obj of targets) {
        if (action.type === "translate") {
          const start = action.from ? new THREE.Vector3(...action.from) : obj.position.clone();
          const end = action.to
            ? new THREE.Vector3(...action.to)
            : action.by
              ? start.clone().add(new THREE.Vector3(...action.by))
              : start;
          obj.position.lerpVectors(start, end, p);
        } else if (action.axis && typeof action.angleDeg === "number") {
          const turn = new THREE.Quaternion().setFromAxisAngle(
            new THREE.Vector3(...action.axis).normalize(),
            THREE.MathUtils.degToRad(action.angleDeg) * p
          );
          obj.quaternion.premultiply(turn); // about the parent's axis, like Vishwa
        }
      }
    } else if ((action.type === "highlight" || action.type === "pulse") && current) {
      const start = action.delay ?? 0;
      const end = actionEnd(action);
      const active = time >= start && (time < end || action.hold === true);
      for (const obj of targets) {
        for (const line of this.outlinesFor(obj, action.color ?? "#ffd600")) {
          line.visible = active;
          const material = line.material as THREE.LineBasicMaterial;
          material.color.set(action.color ?? "#ffd600");
          material.opacity =
            action.type === "pulse" ? 0.35 + 0.65 * Math.abs(Math.sin(((time - start) / 600) * Math.PI)) : 1;
        }
      }
    }
  }

  private outlinesFor(obj: THREE.Object3D, color: string): THREE.LineSegments[] {
    let lines = this.outlines.get(obj);
    if (!lines) {
      lines = [];
      const meshes: THREE.Mesh[] = [];
      obj.traverse((child) => {
        if ((child as THREE.Mesh).isMesh && !child.userData[PROCEDURE_OUTLINE]) meshes.push(child as THREE.Mesh);
      });
      for (const mesh of meshes) {
        // Drawn on top, so the part stays visible inside / behind others.
        const line = new THREE.LineSegments(
          new THREE.EdgesGeometry(mesh.geometry, 25),
          new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true })
        );
        line.renderOrder = 1000;
        line.userData[PROCEDURE_OUTLINE] = true;
        line.visible = false;
        mesh.add(line);
        lines.push(line);
      }
      this.outlines.set(obj, lines);
    }
    return lines;
  }
}

/** What the AR view shows over the video for the current step. */
export interface StepGuide {
  number: number;
  total: number;
  title: string;
  instruction: string | null;
  state: "waiting" | "holding" | "done" | "complete" | "manual";
  statusText: string;
}

type HostEvent =
  | { type: "model" }
  | { type: "checks"; checks: PartCheck[] | null }
  | { type: "guide"; guide: StepGuide | null };

/**
 * Connects whichever view draws the model (3D viewer or AR overlay) with the
 * procedure panel, without re-rendering React on every frame: the view
 * announces its loaded parts, calls `tick` each frame and publishes the
 * camera's part checks; the panel listens, and publishes the current step
 * back for the view to show.
 */
export class ProcedureHost {
  objects = new Map<number, THREE.Object3D>();
  tick: ((now: number) => void) | null = null;
  guide: StepGuide | null = null; // latest, for views that mount after it was published
  private listeners = new Set<(event: HostEvent) => void>();

  attach(objects: Map<number, THREE.Object3D>): void {
    this.objects = objects;
    this.emit({ type: "model" });
  }

  detach(objects: Map<number, THREE.Object3D>): void {
    if (this.objects !== objects) return; // another view already took over
    this.objects = new Map();
    this.emit({ type: "model" });
  }

  publishChecks(checks: PartCheck[] | null): void {
    this.emit({ type: "checks", checks });
  }

  publishGuide(guide: StepGuide | null): void {
    this.guide = guide;
    this.emit({ type: "guide", guide });
  }

  subscribe(listener: (event: HostEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: HostEvent): void {
    this.listeners.forEach((l) => l(event));
  }
}
