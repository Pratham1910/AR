import { useEffect, useState } from "react";
import { useCamera } from "../../hooks/useCamera";
import { InspectionApi } from "../../services/api";
import type { InspectionRun, StepValidationResponse } from "../../types";

interface Props {
  assetId: string;
  procedureRevisionId: string;
  operator?: string;
}

/**
 * The operator-facing runtime loop (Project.md #41, #42, #43):
 *   show expected state -> capture frame -> observe -> validate -> show
 *   PASS/FAIL/UNCERTAIN with reason and evidence -> advance or retry.
 */
export function InspectionRunner({ assetId, procedureRevisionId, operator }: Props) {
  const { videoRef, canvasRef, ready, error: cameraError, captureFrameBase64 } = useCamera();
  const [run, setRun] = useState<InspectionRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [lastResult, setLastResult] = useState<StepValidationResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    InspectionApi.start(assetId, procedureRevisionId, operator)
      .then((res) => InspectionApi.get(res.inspection_run_id))
      .then(setRun)
      .catch((err: Error) => setApiError(err.message));
    // Intentionally run once per (assetId, procedureRevisionId) pair.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assetId, procedureRevisionId]);

  const currentStep = run?.steps.find((s) => s.result === "NOT_EVALUATED") ?? null;
  const isComplete = run !== null && currentStep === null;

  const handleCaptureAndValidate = async () => {
    if (!run || !currentStep) return;
    setBusy(true);
    setApiError(null);
    try {
      const frame = captureFrameBase64();
      if (!frame) throw new Error("Could not capture a frame from the camera.");
      await InspectionApi.observe(run.id, currentStep.step_id, frame);
      const result = await InspectionApi.validate(run.id, currentStep.step_id);
      setLastResult(result);
      const refreshed = await InspectionApi.get(run.id);
      setRun(refreshed);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleComplete = async () => {
    if (!run) return;
    await InspectionApi.complete(run.id);
    const refreshed = await InspectionApi.get(run.id);
    setRun(refreshed);
  };

  return (
    <div className="inspection-runner">
      <div className="camera-pane">
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", maxWidth: 640 }} />
        <canvas ref={canvasRef} style={{ display: "none" }} />
        {!ready && !cameraError && <p>Requesting camera access…</p>}
        {cameraError && <p className="error">Camera error: {cameraError}</p>}
      </div>

      <div className="step-pane">
        {run && !isComplete && currentStep && (
          <>
            <h3>
              Step {run.steps.indexOf(currentStep) + 1} / {run.steps.length}
            </h3>
            <p className="step-title">{currentStep.title}</p>
            <button onClick={handleCaptureAndValidate} disabled={busy || !ready}>
              {busy ? "Validating…" : "Capture & Validate"}
            </button>

            {lastResult && lastResult.stepId === currentStep.step_id_str && (
              <ResultPanel result={lastResult} />
            )}
          </>
        )}

        {isComplete && (
          <div>
            <h3>All steps evaluated.</h3>
            {run?.status !== "completed" && <button onClick={handleComplete}>Complete Inspection</button>}
            {run?.status === "completed" && <p>Inspection run completed.</p>}
            <StepList run={run} />
          </div>
        )}

        {apiError && <p className="error">{apiError}</p>}
      </div>
    </div>
  );
}

function ResultPanel({ result }: { result: StepValidationResponse }) {
  const resultClass = result.result.toLowerCase();
  return (
    <div className={`result-panel result-${resultClass}`}>
      <p>
        EXPECTED: <strong>{result.expectedState}</strong>
      </p>
      <p>
        OBSERVED: <strong>{result.observedState ?? "—"}</strong>
      </p>
      <p>
        VISION: confidence {(result.confidence * 100).toFixed(0)}% — detected{" "}
        {result.validation.objectDetected ? "yes" : "no"}, state matched{" "}
        {result.validation.stateMatched ? "yes" : "no"}
      </p>
      <p className={`result-badge result-${resultClass}`}>{result.result}</p>
      {result.reason && <p className="reason">Reason: {result.reason}</p>}
    </div>
  );
}

function StepList({ run }: { run: InspectionRun | null }) {
  if (!run) return null;
  return (
    <ul>
      {run.steps.map((s) => (
        <li key={s.step_id}>
          {s.step_id_str} — {s.title}: <strong>{s.result}</strong>
        </li>
      ))}
    </ul>
  );
}
