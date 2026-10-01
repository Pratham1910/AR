import { useEffect, useRef, useState } from "react";
import { isAxiosError } from "axios";
import { Models3DApi, apiErrorMessage } from "../../services/api";
import type { Model3DPart, PartCheck, ProcedureSummary } from "../../types";
import {
  expectedVisionFacts,
  factStatus,
  makePartResolver,
  ProcedurePlayer,
  skippedActionTypes,
  stepDuration,
  type FactStatus,
  type ProcedureHost,
  type ProcedurePackage,
} from "./procedure";

interface Props {
  modelId: string;
  parts: Model3DPart[];
  host: ProcedureHost;
  /** True in AR Registration, where part checks come from the camera; elsewhere steps are marked done by hand. */
  cameraChecks: boolean;
}

// Consecutive camera frames a step's end state must hold before it counts as
// done, so one misread frame can't pass (or a flicker reset) a step.
const REQUIRED_FRAMES = 5;
const LOOP_PAUSE_MS = 1200; // the step animation replays after this pause
const ADVANCE_DELAY_MS = 1500; // time to see "done" before the next step starts

const FACT_TEXT: Record<FactStatus, string> = {
  met: "✅ yes",
  not_met: "❌ not yet",
  uncertain: "❓ unsure",
  unknown: "⏳ waiting for tracking",
};

/**
 * A Vishwa procedure (.procedure.json) attached to this model: imports it,
 * plays the current step's animation on the model's parts, and — in AR —
 * marks the step done when the camera sees its expected end state (e.g. the
 * cap absent), then moves to the next step.
 */
export function ProcedurePanel({ modelId, parts, host, cameraChecks }: Props) {
  const [pkg, setPkg] = useState<ProcedurePackage | null>(null);
  const [summary, setSummary] = useState<ProcedureSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [passed, setPassed] = useState<string[]>([]);
  const [checks, setChecks] = useState<PartCheck[] | null>(null);
  const [streak, setStreak] = useState(0);
  const [unmatched, setUnmatched] = useState<string[]>([]);

  const playback = useRef({ index: 0, startedAt: performance.now(), playing: true, frozenAt: 0 });
  const streakRef = useRef(0);
  const advanceTimer = useRef<number | null>(null);

  const procedure = pkg?.procedure ?? null;
  const step = procedure?.steps[index] ?? null;

  useEffect(() => {
    setPkg(null);
    setSummary(null);
    setError(null);
    Models3DApi.procedure(modelId)
      .then((p) => {
        setPkg(p);
        return Models3DApi.procedureSummary(modelId).then(setSummary);
      })
      .catch((err) => {
        if (!(isAxiosError(err) && err.response?.status === 404)) setError(apiErrorMessage(err));
      });
  }, [modelId]);

  const goTo = (i: number) => {
    if (advanceTimer.current !== null) window.clearTimeout(advanceTimer.current);
    advanceTimer.current = null;
    playback.current.index = i;
    playback.current.startedAt = performance.now();
    playback.current.playing = true;
    setPlaying(true);
    streakRef.current = 0;
    setStreak(0);
    setIndex(i);
  };

  // Restart from step 1 whenever a (different) procedure is loaded.
  useEffect(() => {
    setPassed([]);
    goTo(0);
  }, [pkg]);

  // (Re)build the player whenever the procedure, the part names or the drawn model change.
  useEffect(() => {
    if (!procedure) return;
    let player: ProcedurePlayer | null = null;
    const build = () => {
      player?.reset();
      player = host.objects.size > 0 ? new ProcedurePlayer(procedure, makePartResolver(parts, host.objects)) : null;
      setUnmatched(player?.unmatched ?? []);
    };
    build();
    host.tick = (now) => {
      const state = playback.current;
      const current = procedure.steps[state.index];
      if (!player || !current) return;
      const duration = stepDuration(current);
      if (state.playing) state.frozenAt = Math.min((now - state.startedAt) % (duration + LOOP_PAUSE_MS), duration);
      player.seek(state.index, state.frozenAt);
    };
    const unsubscribe = host.subscribe((event) => event.type === "model" && build());
    return () => {
      unsubscribe();
      host.tick = null;
      player?.reset(); // leave the model assembled
    };
  }, [procedure, parts, host]);

  // Camera part checks -> is the current step's end state reached?
  useEffect(() => {
    if (!cameraChecks) return;
    return host.subscribe((event) => {
      if (event.type !== "checks") return;
      setChecks(event.checks);
      const current = procedure?.steps[playback.current.index];
      if (!procedure || !current || advanceTimer.current !== null) return;
      const facts = expectedVisionFacts(procedure, current);
      if (facts.length === 0) return;
      const met = facts.every((f) => factStatus(f, event.checks) === "met");
      streakRef.current = met ? streakRef.current + 1 : 0;
      setStreak(streakRef.current);
      if (streakRef.current >= REQUIRED_FRAMES) complete(playback.current.index);
    });
  }, [cameraChecks, procedure, host]);

  useEffect(() => () => {
    if (advanceTimer.current !== null) window.clearTimeout(advanceTimer.current);
  }, []);

  const complete = (i: number) => {
    if (!procedure) return;
    const id = procedure.steps[i].id;
    setPassed((p) => (p.includes(id) ? p : [...p, id]));
    // Show the step's end pose while "done" is on screen.
    playback.current.startedAt = performance.now() - stepDuration(procedure.steps[i]);
    playback.current.playing = false;
    playback.current.frozenAt = stepDuration(procedure.steps[i]);
    setPlaying(false);
    if (i + 1 < procedure.steps.length) {
      advanceTimer.current = window.setTimeout(() => {
        advanceTimer.current = null;
        goTo(i + 1);
      }, ADVANCE_DELAY_MS);
    }
  };

  const togglePlay = () => {
    const state = playback.current;
    state.playing = !state.playing;
    if (state.playing) state.startedAt = performance.now() - state.frozenAt;
    setPlaying(state.playing);
  };

  const importFile = async (file: File) => {
    setError(null);
    try {
      const json = JSON.parse(await file.text());
      const result = await Models3DApi.uploadProcedure(modelId, json);
      setSummary(result);
      setPkg(await Models3DApi.procedure(modelId));
    } catch (err) {
      setError(err instanceof SyntaxError ? `Not valid JSON: ${err.message}` : apiErrorMessage(err));
    }
  };

  const remove = async () => {
    await Models3DApi.removeProcedure(modelId);
    setPkg(null);
    setSummary(null);
  };

  const facts = procedure && step ? expectedVisionFacts(procedure, step) : [];
  const allDone = procedure !== null && passed.length === procedure.steps.length;
  const skipped = procedure ? skippedActionTypes(procedure) : [];

  return (
    <div className="procedure-panel">
      <h4>Procedure</h4>
      {error && <p className="error">{error}</p>}
      {!procedure && (
        <p className="hint">
          Import a procedure exported from Vishwa (<code>.procedure.json</code>). Its part names are matched to the
          names in the Parts panel.
        </p>
      )}
      <div className="parts-actions">
        <label className="file-button">
          {procedure ? "Replace…" : "Import .procedure.json"}
          <input
            type="file"
            accept=".json,application/json"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void importFile(file);
            }}
          />
        </label>
        {procedure && (
          <button className="danger" onClick={() => void remove()}>
            Remove
          </button>
        )}
      </div>

      {procedure && step && (
        <>
          <p>
            <strong>{procedure.title ?? procedure.id}</strong>
            {summary && (
              <span className="hint">
                {" "}
                · {summary.steps} steps, {summary.actions} actions
              </span>
            )}
          </p>
          {unmatched.length > 0 && (
            <p className="warning">
              Not found in this model: {unmatched.join(", ")} — name the parts in the Parts panel to match.
            </p>
          )}
          {skipped.length > 0 && <p className="hint">Not played here: {skipped.join(", ")}</p>}

          <ol className="procedure-steps">
            {procedure.steps.map((s, i) => (
              <li key={s.id} className={i === index ? "current" : ""} onClick={() => goTo(i)}>
                {passed.includes(s.id) ? "✅" : i === index ? "▶" : "○"} {s.title ?? s.id}
              </li>
            ))}
          </ol>

          <div className="procedure-step">
            <div className="step-title">
              Step {index + 1}/{procedure.steps.length}: {step.title ?? step.id}
            </div>
            {step.description && <p>{step.description}</p>}
            {facts.length > 0 && (
              <div className="procedure-checks">
                {facts.map((f) => (
                  <div key={`${f.kind}-${f.part}`}>
                    {f.part} {f.kind === "partAbsent" ? "removed" : "in place"}?{" "}
                    {cameraChecks ? (
                      <strong>{FACT_TEXT[factStatus(f, checks)]}</strong>
                    ) : (
                      <span className="hint">checked by the camera in AR Registration</span>
                    )}
                  </div>
                ))}
                {cameraChecks && !passed.includes(step.id) && streak > 0 && (
                  <div className="hint">
                    holding… {Math.min(streak, REQUIRED_FRAMES)}/{REQUIRED_FRAMES} frames
                  </div>
                )}
              </div>
            )}
            {passed.includes(step.id) && <p className="procedure-done">✅ Step done</p>}
            {allDone && <p className="procedure-done">🎉 Procedure complete</p>}
          </div>

          <div className="parts-actions">
            <button disabled={index === 0} onClick={() => goTo(index - 1)}>
              ◀ Back
            </button>
            <button onClick={togglePlay}>{playing ? "⏸ Pause" : "▶ Play"}</button>
            <button onClick={() => goTo(index)}>↺ Replay</button>
            {(facts.length === 0 || !cameraChecks) && !passed.includes(step.id) && (
              <button onClick={() => complete(index)}>Mark done</button>
            )}
            <button disabled={index + 1 >= procedure.steps.length} onClick={() => goTo(index + 1)}>
              Next ▶
            </button>
          </div>
        </>
      )}
    </div>
  );
}
