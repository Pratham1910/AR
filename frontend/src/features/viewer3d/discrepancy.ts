import * as THREE from "three";
import type { ComponentResult, ComponentStatus, RegisteredComponent } from "../../types";

/*
 * Draws component inspection results onto the AR overlay, separately from
 * the normal model rendering:
 *
 *   MISSING_CONFIRMED  red ring + "MISSING: <ID>" + a translucent red ghost of the
 *                      part, whose normal opaque mesh is hidden so the reference
 *                      model doesn't look installed where the real part is missing
 *   MISSING_CANDIDATE  amber ring ("checking")
 *   PRESENT            green ring            (only with "show all")
 *   UNCERTAIN/OCCLUDED neutral grey ring     (only with "show all")
 *
 * Markers live inside the overlay's model group, at each component's centre
 * in that group's frame (the backend registry's center_m: recentred, metres),
 * so they follow the tracked object with no pose maths of their own. Rings
 * are at least MIN_RING_M across so a missing 3 mm screw is still visible.
 * Everything created here is disposed when it's removed; hidden meshes are
 * restored on clear().
 */

export const STATUS_COLOR: Record<ComponentStatus, string> = {
  PRESENT: "#5fe39a",
  MISSING_CANDIDATE: "#ffb020",
  MISSING_CONFIRMED: "#ff5c5c",
  UNCERTAIN: "#9e9e9e",
  OCCLUDED: "#9e9e9e",
  NOT_INSPECTED: "#9e9e9e",
};

/** Colours that tell components apart. No reds or greens: those mean missing / present. */
export const COMPONENT_PALETTE = ["#4aa3ff", "#c58aff", "#2ec4b6", "#ff8fab", "#ffd166", "#22d3ee", "#a78bfa", "#f472b6", "#60a5fa", "#fbbf24"];

/** A component's own colour: the one chosen in its mapping, else one from the palette by its place in the registry. */
export function componentColor(component: Pick<RegisteredComponent, "color"> | undefined, index: number): string {
  return component?.color ?? COMPONENT_PALETTE[((index % COMPONENT_PALETTE.length) + COMPONENT_PALETTE.length) % COMPONENT_PALETTE.length];
}

const MIN_RING_M = 0.025;
const RING_GROWTH = 1.6; // ring diameter vs the component's largest dimension
const LABEL_HEIGHT_M = 0.012;
const SELECTED_GROWTH = 1.35;
const MARKER_KEY = "tvastaDiscrepancy";

/** Makes a texture for a ring or a text label (a canvas in the browser; a stub in tests). */
export interface MarkerTextures {
  ring(color: string, emphasis: boolean): THREE.Texture;
  label(text: string, color: string): { texture: THREE.Texture; aspect: number };
}

export interface DiscrepancyOptions {
  showAll: boolean; // also mark present / uncertain / occluded components
  selected: string | null; // highlighted from the results panel, whatever its status
}

interface Marker {
  key: string; // what it was built for: rebuilt when this changes
  group: THREE.Group;
  ghosts: THREE.Mesh[];
  hidden: THREE.Mesh[]; // original meshes suppressed while this marker exists
}

const markerLabel = (status: ComponentStatus, id: string) =>
  status === "MISSING_CONFIRMED"
    ? `MISSING: ${id}`
    : status === "MISSING_CANDIDATE"
      ? `CHECKING: ${id}`
      : status === "PRESENT"
        ? `OK: ${id}`
        : `${status.replace("_", " ")}: ${id}`;

export class DiscrepancyRenderer {
  private readonly markers = new Map<string, Marker>();
  private components = new Map<string, RegisteredComponent>();

  private readonly root: THREE.Object3D;
  private readonly parts: Map<number, THREE.Object3D>;
  private readonly textures: MarkerTextures;

  constructor(root: THREE.Object3D, parts: Map<number, THREE.Object3D>, textures: MarkerTextures) {
    this.root = root;
    this.parts = parts;
    this.textures = textures;
  }

  /** The registry (center_m, size_m, node_index) the results refer to. */
  setComponents(components: RegisteredComponent[]): void {
    this.components = new Map(components.map((c) => [c.id, c]));
    this.clear(); // positions may have changed
  }

  /** Brings the markers in line with the latest results. */
  update(results: ComponentResult[], options: DiscrepancyOptions): void {
    const wanted = new Map<string, string>();
    for (const result of results) {
      const component = this.components.get(result.id);
      if (!component?.center_m) continue;
      const selected = options.selected === result.id;
      const flagged = result.status === "MISSING_CONFIRMED" || result.status === "MISSING_CANDIDATE";
      if (!flagged && !options.showAll && !selected) continue;
      wanted.set(result.id, `${result.status}|${selected}`);
    }
    for (const [id, marker] of this.markers) {
      if (wanted.get(id) !== marker.key) this.remove(id);
    }
    for (const [id, key] of wanted) {
      if (this.markers.has(id)) continue;
      const [status, selected] = key.split("|") as [ComponentStatus, string];
      this.markers.set(id, this.build(this.components.get(id)!, status, selected === "true", key));
    }
  }

  /** Removes every marker and restores hidden meshes (inspection off / reset). */
  clear(): void {
    for (const id of [...this.markers.keys()]) this.remove(id);
  }

  dispose(): void {
    this.clear();
  }

  /** For tests and debugging: what is drawn right now. */
  get markerIds(): string[] {
    return [...this.markers.keys()];
  }

  private build(component: RegisteredComponent, status: ComponentStatus, selected: boolean, key: string): Marker {
    const color = STATUS_COLOR[status];
    const group = new THREE.Group();
    group.name = `inspection:${component.id}`;
    group.userData[MARKER_KEY] = true;
    const [x, y, z] = component.center_m!;
    group.position.set(x, y, z);

    const size = Math.max(...(component.size_m ?? [0]));
    const diameter = Math.max(MIN_RING_M, size * RING_GROWTH) * (selected ? SELECTED_GROWTH : 1);
    const ring = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: this.textures.ring(color, selected), depthTest: false, transparent: true })
    );
    ring.scale.set(diameter, diameter, 1);
    ring.renderOrder = 1000;
    group.add(ring);

    const { texture, aspect } = this.textures.label(markerLabel(status, component.id), color);
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
    label.scale.set(LABEL_HEIGHT_M * aspect, LABEL_HEIGHT_M, 1);
    label.center.set(0, 0.5); // anchored at its left edge, beside the ring
    label.position.set(diameter / 2 + 0.004, 0, 0);
    label.renderOrder = 1001;
    group.add(label);

    const ghosts: THREE.Mesh[] = [];
    const hidden: THREE.Mesh[] = [];
    // A whole-node component confirmed missing: show where it belongs as a
    // red ghost and hide the opaque reference part. (A body of a combined
    // mesh has no mesh of its own here: the ring marks it.)
    const part = component.piece === null && component.node_index !== null ? this.parts.get(component.node_index) : undefined;
    if (status === "MISSING_CONFIRMED" && part) {
      const meshes: THREE.Mesh[] = [];
      part.traverse((child) => {
        if ((child as THREE.Mesh).isMesh && !child.userData[MARKER_KEY]) meshes.push(child as THREE.Mesh);
      });
      for (const mesh of meshes) {
        const ghost = new THREE.Mesh(
          mesh.geometry, // shared with the part: not disposed here
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.4, depthTest: false, depthWrite: false })
        );
        ghost.userData[MARKER_KEY] = true;
        ghost.renderOrder = 999;
        mesh.add(ghost); // follows the part's own transform (and any procedure animation)
        ghosts.push(ghost);
        mesh.layers.disable(0); // not drawn by the overlay camera; its children (the ghost) still are
        hidden.push(mesh);
      }
    }
    this.root.add(group);
    return { key, group, ghosts, hidden };
  }

  private remove(id: string): void {
    const marker = this.markers.get(id);
    if (!marker) return;
    marker.group.removeFromParent();
    marker.group.traverse((child) => {
      const material = (child as THREE.Sprite).material as THREE.SpriteMaterial | undefined;
      if (material) {
        material.map?.dispose();
        material.dispose();
      }
    });
    for (const ghost of marker.ghosts) {
      ghost.removeFromParent();
      (ghost.material as THREE.Material).dispose();
    }
    for (const mesh of marker.hidden) mesh.layers.enable(0);
    this.markers.delete(id);
  }
}

/** Ring and label textures drawn on canvases (browser). */
export const canvasMarkerTextures: MarkerTextures = {
  ring(color, emphasis) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext("2d")!;
    ctx.lineWidth = emphasis ? 14 : 10;
    ctx.strokeStyle = "rgba(0,0,0,0.6)"; // dark halo: readable on any background
    ctx.beginPath();
    ctx.arc(64, 64, 52, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = emphasis ? 9 : 6;
    ctx.strokeStyle = emphasis ? "#ffffff" : color;
    ctx.stroke();
    if (emphasis) {
      ctx.lineWidth = 5;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.arc(64, 64, 40, 0, Math.PI * 2);
      ctx.stroke();
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  },
  label(text, color) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const font = "bold 44px sans-serif";
    ctx.font = font;
    const width = Math.ceil(ctx.measureText(text).width) + 24;
    canvas.width = width;
    canvas.height = 64;
    ctx.font = font;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(0, 0, width, 64);
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, 12, 34);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return { texture, aspect: width / 64 };
  },
};
