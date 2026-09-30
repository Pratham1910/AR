"""
FBX -> GLB conversion at upload time, via headless Blender.

Everything downstream (Three.js viewers/overlays, glb_inspect's scale
computation, the MegaPose pose service) speaks GLB, so FBX is converted once
here rather than supported separately in each of those places.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

_SCRIPT = Path(__file__).with_name("blender_fbx_to_glb.py")

_DEFAULT_BLENDER_LOCATIONS = sorted(
    Path("C:/Program Files/Blender Foundation").glob("Blender */blender.exe"), reverse=True
)


class FbxConversionError(RuntimeError):
    pass


class BlenderNotFound(FbxConversionError):
    pass


def find_blender(configured_path: str | None) -> Path | None:
    if configured_path:
        path = Path(configured_path)
        return path if path.exists() else None
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    return _DEFAULT_BLENDER_LOCATIONS[0] if _DEFAULT_BLENDER_LOCATIONS else None


def convert_fbx_to_glb(fbx_bytes: bytes, blender_path: str | None, timeout_s: float = 180.0) -> bytes:
    blender = find_blender(blender_path)
    if blender is None:
        raise BlenderNotFound(
            "FBX upload needs Blender to convert it to GLB, but Blender wasn't found. "
            "Install Blender or set BLENDER_PATH in .env."
        )

    with tempfile.TemporaryDirectory(prefix="tvasta-fbx-") as tmp:
        src = Path(tmp) / "model.fbx"
        dst = Path(tmp) / "model.glb"
        src.write_bytes(fbx_bytes)
        try:
            result = subprocess.run(
                [str(blender), "-b", "--factory-startup", "--python", str(_SCRIPT), "--", str(src), str(dst)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise FbxConversionError(f"Blender took longer than {timeout_s:.0f}s converting this FBX") from exc

        if result.returncode != 0 or not dst.exists():
            output = (result.stderr or "") + (result.stdout or "")
            marker = next((l for l in output.splitlines() if "TVASTA_CONVERT_ERROR" in l), None)
            detail = marker.split(":", 1)[1].strip() if marker else output.strip()[-500:]
            raise FbxConversionError(f"Could not convert this FBX: {detail}")
        return dst.read_bytes()
