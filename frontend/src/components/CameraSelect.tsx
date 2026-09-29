import type { CameraDeviceOption } from "../hooks/useCamera";

interface Props {
  devices: CameraDeviceOption[];
  selectedDeviceId: string | null;
  onSelect: (deviceId: string) => void;
}

/**
 * Lets the operator pick which physical camera to use — important on
 * machines with more than one video input (a laptop's real webcam plus an
 * IR/depth camera, a virtual camera from other software, etc.), where the
 * browser's default choice is sometimes not the one you want and can show
 * as a black frame.
 */
export function CameraSelect({ devices, selectedDeviceId, onSelect }: Props) {
  if (devices.length <= 1) return null;

  return (
    <label className="camera-select">
      Camera
      <select value={selectedDeviceId ?? ""} onChange={(e) => onSelect(e.target.value)}>
        {devices.map((d) => (
          <option key={d.deviceId} value={d.deviceId}>
            {d.label}
          </option>
        ))}
      </select>
    </label>
  );
}
