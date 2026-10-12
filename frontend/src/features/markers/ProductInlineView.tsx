import { useEffect, useMemo, useState } from "react";
import { Models3DApi, resolveApiUrl } from "../../services/api";
import type { Model3DPart, ScanProduct } from "../../types";
import { EMPTY_PART_VIEW, type PartView } from "../viewer3d/parts";
import { ProcedureHost } from "../viewer3d/procedure";
import { ProcedurePanel } from "../viewer3d/ProcedurePanel";
import { ThreeViewer } from "../viewer3d/ThreeViewer";

export type ProductView = "model" | "procedure";

interface Props {
  product: ScanProduct;
  view: ProductView;
  /** Opens the Procedure inspection tab on a published revision — only ever on an explicit click. */
  onOpenInspection?: (assetId: string, revisionId: string) => void;
}

/**
 * A scanned product's 3D model or procedure, shown in place under the product
 * scan — nothing navigates, the scanner stays where it is.
 *
 * Both views are the application's existing components, unchanged: the model
 * is ThreeViewer, the procedure is ProcedurePanel. They are linked the same
 * way the 3D workspace links them — through a ProcedureHost — but with a host
 * of their own, so playing a step here never moves the model in the AR view
 * above (which may even be a different product's).
 *
 * The procedure player animates the model's parts, so the procedure view
 * keeps the model on screen and adds the player beneath it; switching views
 * therefore never reloads the model.
 */
export function ProductInlineView({ product, view, onOpenInspection }: Props) {
  const modelId = product.model_id;
  const host = useMemo(() => new ProcedureHost(), [modelId]);
  const [parts, setParts] = useState<Model3DPart[]>([]);
  const [partView, setPartView] = useState<PartView>(EMPTY_PART_VIEW);

  useEffect(() => {
    setParts([]);
    setPartView(EMPTY_PART_VIEW);
    if (!modelId) return;
    let stale = false;
    Models3DApi.parts(modelId)
      .then((list) => !stale && setParts(list))
      .catch(() => undefined); // a single-part model simply has no parts
    return () => {
      stale = true;
    };
  }, [modelId]);

  const inspections = product.procedures;
  const showPlayer = view === "procedure" && product.has_model_procedure && modelId !== null;
  // The model is on screen for the model view, and for the procedure view when there is a procedure to play on it.
  const showModel = modelId !== null && product.model_url !== null && (view === "model" || showPlayer);

  return (
    <div className="card product-inline">
      <div className="panel-title">
        {view === "model" ? "3D model" : "Procedure"} — {product.asset_name}
        {showModel && product.model_name && <span className="hint">{product.model_name}</span>}
      </div>

      {view === "model" && !showModel && (
        <p className="warning">
          No 3D model is registered for {product.asset_name}. Upload one in 3D &amp; AR workspace → “+ Upload 3D model”.
        </p>
      )}

      {view === "procedure" && !product.has_model_procedure && (
        <p className={inspections.length > 0 ? "hint" : "warning"}>
          {modelId
            ? `No procedure is attached to ${product.asset_name}'s 3D model.`
            : `${product.asset_name} has no 3D model, so no model procedure can be attached to it.`}
        </p>
      )}

      {showModel && (
        <ThreeViewer
          key={modelId}
          modelUrl={resolveApiUrl(product.model_url!)}
          height={420}
          partView={partView}
          onPartClick={(nodeIndex) => setPartView((v) => ({ ...v, highlight: nodeIndex }))}
          procedureHost={host}
        />
      )}

      {showPlayer && (
        <div className="product-inline-procedure">
          <ProcedurePanel key={modelId} modelId={modelId!} parts={parts} host={host} cameraChecks={false} />
        </div>
      )}

      {view === "procedure" && inspections.length > 0 && (
        <div className="product-inline-inspections">
          <div className="scan-key">Inspection procedures</div>
          <ul>
            {inspections.map((p) => (
              <li key={p.procedure_id}>
                <strong>{p.title}</strong> <span className="hint">({p.procedure_id_str})</span>{" "}
                {p.published_revision_id ? (
                  onOpenInspection && (
                    <button onClick={() => onOpenInspection(product.asset_id, p.published_revision_id!)}>
                      Run in Procedure inspection →
                    </button>
                  )
                ) : (
                  <span className="hint">no published revision</span>
                )}
              </li>
            ))}
          </ul>
          <p className="hint">
            An inspection procedure is run step by step against the camera, with its own camera view, so it opens in the
            Procedure inspection tab — only when you press the button.
          </p>
        </div>
      )}
    </div>
  );
}
