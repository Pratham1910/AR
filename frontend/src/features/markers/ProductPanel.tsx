import { useEffect, useMemo, useState } from "react";
import { Models3DApi, resolveApiUrl } from "../../services/api";
import type { Model3DPart, TagProduct } from "../../types";
import { ProcedurePanel } from "../viewer3d/ProcedurePanel";
import { ProcedureHost } from "../viewer3d/procedure";
import { ThreeViewer } from "../viewer3d/ThreeViewer";

interface Props {
  product: TagProduct;
  /** Opens the Procedure inspection tab for one of the product's inspection procedures. */
  onOpenInspection?: (assetId: string, revisionId: string) => void;
  onClose?: () => void;
}

type View = "none" | "model" | "procedure";

/**
 * A product identified by its tags, with its 3D model and procedure opened
 * in place — in the application's own viewer (ThreeViewer) and procedure
 * player (ProcedurePanel), not copies of them. The procedure player gets its
 * own ProcedureHost, so a step played here never moves the model in the AR
 * view above.
 */
export function ProductPanel({ product, onOpenInspection, onClose }: Props) {
  const [view, setView] = useState<View>("none");
  const [parts, setParts] = useState<Model3DPart[]>([]);
  const host = useMemo(() => new ProcedureHost(), [product.asset_id]);
  const procedureModel = product.procedure_model;

  useEffect(() => setView("none"), [product.asset_id]);
  useEffect(() => {
    setParts([]);
    if (view !== "procedure" || !procedureModel) return;
    Models3DApi.parts(procedureModel.id)
      .then(setParts)
      .catch(() => setParts([]));
  }, [view, procedureModel?.id]);

  const hasProcedure = procedureModel !== null || product.procedures.length > 0;
  const modelReason = product.model ? undefined : "No 3D model uploaded for this product (3D & AR workspace → + Upload 3D model).";
  const procedureReason = hasProcedure ? undefined : "No procedure for this product: attach one to its 3D model, or create an inspection procedure.";

  return (
    <div className="card product-panel">
      <div className="product-head">
        <div>
          <div className="panel-title">{product.asset_name}</div>
          {product.description && <p className="hint">{product.description}</p>}
          <p className="hint">
            Object class: {product.detection_class ?? "not set (any object between the tags is accepted)"}
          </p>
        </div>
        {onClose && (
          <button className="link" onClick={onClose}>
            Close
          </button>
        )}
      </div>

      <div className="product-actions">
        <button
          className={view === "model" ? "primary" : ""}
          disabled={!product.model}
          title={modelReason}
          onClick={() => setView((v) => (v === "model" ? "none" : "model"))}
        >
          {view === "model" ? "Hide 3D model" : "View 3D model"}
        </button>
        <button
          className={view === "procedure" ? "primary" : ""}
          disabled={!hasProcedure}
          title={procedureReason}
          onClick={() => setView((v) => (v === "procedure" ? "none" : "procedure"))}
        >
          {view === "procedure" ? "Hide procedure" : "View procedure"}
        </button>
      </div>
      {(modelReason || procedureReason) && (
        <ul className="hint product-missing">
          {modelReason && <li>{modelReason}</li>}
          {procedureReason && <li>{procedureReason}</li>}
        </ul>
      )}

      {view === "model" && product.model && (
        <div className="product-inline">
          <ThreeViewer key={product.model.id} modelUrl={resolveApiUrl(product.model.url)} height={420} />
        </div>
      )}

      {view === "procedure" && (
        <div className="product-inline">
          {procedureModel ? (
            <div className="viewer-with-parts">
              <div className="viewer-main">
                <ThreeViewer key={procedureModel.id} modelUrl={resolveApiUrl(procedureModel.url)} height={420} procedureHost={host} />
              </div>
              <aside className="side-panels">
                <ProcedurePanel key={procedureModel.id} modelId={procedureModel.id} parts={parts} host={host} cameraChecks={false} />
              </aside>
            </div>
          ) : (
            <p className="hint">No procedure is attached to this product's 3D model.</p>
          )}

          {product.procedures.length > 0 && (
            <div className="product-inspections">
              <div className="section-label">Inspection procedures</div>
              {product.procedures.map((p) => (
                <div key={p.id} className="product-inspection">
                  <span>
                    {p.title} <span className="hint">({p.procedure_id_str}{p.published_revision_id ? `, ${p.step_count} steps` : ""})</span>
                  </span>
                  <button
                    disabled={!p.published_revision_id || !onOpenInspection}
                    title={p.published_revision_id ? "Runs it against the camera in the Procedure inspection tab" : "No published revision yet"}
                    onClick={() => onOpenInspection?.(product.asset_id, p.published_revision_id!)}
                  >
                    Run inspection
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
