import * as THREE from "three";

/*
 * The guide outline's pose as the operator moves it: T_camera_object in the
 * OpenCV camera frame (x right, y down, z forward, metres), the frame the
 * backend uses. Kept as a THREE.Matrix4 for the maths.
 */

/** Row-major 4x4 (as the backend sends poses) -> THREE.Matrix4. */
export const rowsToMatrix = (rows: number[][]) =>
  new THREE.Matrix4().set(...(rows.flat() as Parameters<THREE.Matrix4["set"]>));

/** THREE.Matrix4 -> row-major 4x4 for the backend. */
export const matrixToRows = (m: THREE.Matrix4) => [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => m.elements[c * 4 + r]));

/** Three.js position + quaternion of a model pose given in the OpenCV camera
 *  frame — the same conversion as the backend's cv_model_pose_to_threejs
 *  (rotation C·R, position C·t, C = diag(1, -1, -1)). */
export const toThreePose = (tCameraObject: THREE.Matrix4) => {
  const gl = tCameraObject.clone().premultiply(new THREE.Matrix4().makeScale(1, -1, -1));
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  gl.decompose(position, quaternion, new THREE.Vector3());
  return {
    position: { x: position.x, y: position.y, z: position.z },
    quaternion: { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w },
  };
};

/** Slide across the image by (du, dv) frame pixels, at the object's depth. */
export const slide = (m: THREE.Matrix4, du: number, dv: number, fx: number, fy: number) => {
  const e = m.elements;
  e[12] += (du * e[14]) / fx;
  e[13] += (dv * e[14]) / fy;
};

/** Nearer / farther along the viewing ray (stays on the same image spot). */
export const dolly = (m: THREE.Matrix4, factor: number) => {
  m.elements[12] *= factor;
  m.elements[13] *= factor;
  m.elements[14] *= factor;
};

export type ModelAxis = "x" | "y" | "z";
const AXIS: Record<ModelAxis, THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};

/** Rotate about one of the model's OWN axes (through its centre). */
export const rotateAbout = (m: THREE.Matrix4, axis: ModelAxis, degrees: number) =>
  m.multiply(new THREE.Matrix4().makeRotationAxis(AXIS[axis], THREE.MathUtils.degToRad(degrees)));

/** Turn about the model's own up (+Y) axis. */
export const turn = (m: THREE.Matrix4, degrees: number) => rotateAbout(m, "y", degrees);

/** The model's origin and axis tips (length in metres) in image pixels, for drawing its axes; null if behind the camera. */
export const projectAxes = (
  m: THREE.Matrix4,
  k: { fx: number; fy: number; cx: number; cy: number },
  length: number
): Record<"o" | ModelAxis, [number, number]> | null => {
  const project = (p: THREE.Vector3): [number, number] | null => {
    const c = p.clone().applyMatrix4(m);
    return c.z > 1e-3 ? [k.fx * (c.x / c.z) + k.cx, k.fy * (c.y / c.z) + k.cy] : null;
  };
  const o = project(new THREE.Vector3());
  const x = project(AXIS.x.clone().multiplyScalar(length));
  const y = project(AXIS.y.clone().multiplyScalar(length));
  const z = project(AXIS.z.clone().multiplyScalar(length));
  return o && x && y && z ? { o, x, y, z } : null;
};

/** Tilt about the camera's x axis through the model's centre. */
export const tilt = (m: THREE.Matrix4, degrees: number) => {
  const e = m.elements;
  const [x, y, z] = [e[12], e[13], e[14]];
  m.premultiply(new THREE.Matrix4().makeTranslation(-x, -y, -z))
    .premultiply(new THREE.Matrix4().makeRotationX(THREE.MathUtils.degToRad(degrees)))
    .premultiply(new THREE.Matrix4().makeTranslation(x, y, z));
};
