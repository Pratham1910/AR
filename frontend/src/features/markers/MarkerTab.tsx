import { useEffect, useState } from "react";
import type { PoseResponse, ScanProduct } from "../../types";
import { ProductScanPanel } from "./ProductScanPanel";
import { TagTrackingPanel } from "./TagTrackingPanel";

/** What the AR view (RegistrationOverlay) hands its ArUco marker tab. */
export interface MarkerTabCamera {
  captureFrame: () => string | null;
  ready: boolean;
  stage: HTMLElement | null;
  liveTracking: boolean;
  setLiveTracking: (on: boolean) => void;
  showMarkerPose: (pose: PoseResponse | null) => void;
}

interface Props {
  camera: MarkerTabCamera;
  modelId: string;
  /** Opens the Procedure inspection tab (an explicit button in the product's procedure view). */
  onOpenInspection?: (assetId: string, revisionId: string) => void;
  /** Live tracking confirmed a product whose model isn't loaded. */
  onTrackOtherProduct: (product: ScanProduct) => void;
}

/**
 * The ArUco marker tab's two modes, and the rule that only one runs:
 *
 *   Product scan   any one tag  -> product -> its 3D model / procedure, shown below the scanner
 *   Live tracking  four tags + YOLO inside them -> confirmed -> model overlaid
 *
 * They share the camera feed (one <video>, owned by the AR view) and nothing
 * else: separate panels, separate backend endpoints, separate sessions.
 * Starting either stops the other, so there is never more than one detection
 * loop sending frames.
 */
export function MarkerTab({ camera, modelId, onOpenInspection, onTrackOtherProduct }: Props) {
  const [scanning, setScanning] = useState(false);

  // Live tracking was switched on (the AR view's own button): the scan yields.
  useEffect(() => {
    if (camera.liveTracking) setScanning(false);
  }, [camera.liveTracking]);

  const startScan = () => {
    camera.setLiveTracking(false);
    setScanning(true);
  };

  return (
    <>
      <ProductScanPanel
        captureFrame={camera.captureFrame}
        ready={camera.ready}
        stage={camera.stage}
        scanning={scanning && !camera.liveTracking}
        onStart={startScan}
        onStop={() => setScanning(false)}
        onOpenInspection={onOpenInspection}
      />
      <TagTrackingPanel
        captureFrame={camera.captureFrame}
        stage={camera.stage}
        active={camera.liveTracking}
        modelId={modelId}
        showMarkerPose={camera.showMarkerPose}
        onOtherProduct={onTrackOtherProduct}
      />
    </>
  );
}
