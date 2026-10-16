// Moving the guide outline (src/features/viewer3d/guidePose.ts): the poses sent to the backend.
import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { dolly, matrixToRows, projectAxes, rotateAbout, rowsToMatrix, slide, tilt, toThreePose, turn } from "../src/features/viewer3d/guidePose.ts";

const K = { fx: 900, fy: 900, cx: 640, cy: 360 };
const ROWS = [[1, 0, 0, 0.02], [0, 0, 1, -0.01], [0, -1, 0, 0.4], [0, 0, 0, 1]];
const pose = () => rowsToMatrix(ROWS);
const project = (m: THREE.Matrix4, p = new THREE.Vector3()) => {
  const c = p.clone().applyMatrix4(m); // OpenCV camera frame
  return [K.fx * (c.x / c.z) + K.cx, K.fy * (c.y / c.z) + K.cy];
};
const close = (a: number, b: number, tol = 1e-6) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test("row-major rows survive the round trip", () => {
  assert.deepEqual(matrixToRows(rowsToMatrix(ROWS)), ROWS);
});

test("the display pose matches the backend's OpenCV -> Three.js conversion", () => {
  // Identity 0.5 m ahead: backend gives position (0, 0, -0.5), rotation diag(1,-1,-1) = quaternion (1, 0, 0, 0).
  const { position, quaternion } = toThreePose(rowsToMatrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]]));
  close(position.z, -0.5);
  close(Math.abs(quaternion.x), 1);
  close(quaternion.w, 0);
});

test("dragging moves the outline's centre by exactly the dragged pixels, at the same depth", () => {
  const m = pose();
  const [u0, v0] = project(m);
  slide(m, 30, -12, K.fx, K.fy);
  const [u1, v1] = project(m);
  close(u1 - u0, 30);
  close(v1 - v0, -12);
  close(m.elements[14], 0.4);
});

test("the wheel moves it nearer without moving it in the image", () => {
  const m = pose();
  const [u0, v0] = project(m);
  dolly(m, 0.8);
  const [u1, v1] = project(m);
  close(u1, u0);
  close(v1, v0);
  close(m.elements[14], 0.32);
});

test("turning keeps the centre and the model's up axis, and four quarter turns come back", () => {
  const m = pose();
  const up = new THREE.Vector3(0, 1, 0).transformDirection(m);
  const centre = project(m);
  turn(m, 90);
  const after = project(m);
  close(after[0], centre[0]);
  close(after[1], centre[1]);
  close(up.dot(new THREE.Vector3(0, 1, 0).transformDirection(m)), 1);
  turn(m, 270);
  matrixToRows(m).flat().forEach((v, i) => close(v, ROWS.flat()[i]));
});

test("tilting keeps the centre in place", () => {
  const m = pose();
  const centre = project(m);
  tilt(m, 15);
  const after = project(m);
  close(after[0], centre[0]);
  close(after[1], centre[1]);
});

test("rotating about each of the model's own axes keeps that axis and the centre", () => {
  for (const axis of ["x", "y", "z"] as const) {
    const m = pose();
    const unit = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) }[axis];
    const before = unit.clone().transformDirection(m);
    const centre = project(m);
    rotateAbout(m, axis, 37);
    close(before.dot(unit.clone().transformDirection(m)), 1); // that axis didn't move
    const after = project(m);
    close(after[0], centre[0]);
    close(after[1], centre[1]);
    rotateAbout(m, axis, -37);
    matrixToRows(m).flat().forEach((v, i) => close(v, ROWS.flat()[i]));
  }
});

test("the drawn axes start at the model's centre and point along its axes", () => {
  const m = pose();
  const axes = projectAxes(m, K, 0.05);
  assert.ok(axes);
  const centre = project(m);
  close(axes.o[0], centre[0]);
  close(axes.o[1], centre[1]);
  const xTip = project(m, new THREE.Vector3(0.05, 0, 0));
  close(axes.x[0], xTip[0]);
  close(axes.x[1], xTip[1]);
});
