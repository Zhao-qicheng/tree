export const HUMAN17_JOINTS = Object.freeze([
  "root",
  "right_hip",
  "right_knee",
  "right_ankle",
  "left_hip",
  "left_knee",
  "left_ankle",
  "spine",
  "thorax",
  "nose",
  "head",
  "left_shoulder",
  "left_elbow",
  "left_wrist",
  "right_shoulder",
  "right_elbow",
  "right_wrist",
]);

export const HUMAN17_BONES = Object.freeze([
  [0, 1], [1, 2], [2, 3],
  [0, 4], [4, 5], [5, 6],
  [0, 7], [7, 8], [8, 9], [9, 10],
  [8, 11], [11, 12], [12, 13],
  [8, 14], [14, 15], [15, 16],
]);

export const DEFAULT_PALETTE = Object.freeze({
  center: "#4f83b7",
  left: "#2d9d68",
  right: "#d85857",
  head: "#5b7085",
  skeleton: "#172033",
  root: "#e59e32",
});

const EPSILON = 1e-6;

export function isFinitePoint(value) {
  return Array.isArray(value)
    && value.length >= 3
    && Number.isFinite(Number(value[0]))
    && Number.isFinite(Number(value[1]))
    && Number.isFinite(Number(value[2]));
}

export function isHuman17Pose(value) {
  return Array.isArray(value) && value.length === HUMAN17_JOINTS.length;
}

export function pointDistance(a, b) {
  if (!isFinitePoint(a) || !isFinitePoint(b)) return null;
  const dx = Number(a[0]) - Number(b[0]);
  const dy = Number(a[1]) - Number(b[1]);
  const dz = Number(a[2]) - Number(b[2]);
  const distance = Math.hypot(dx, dy, dz);
  return distance > EPSILON ? distance : null;
}

export function median(values) {
  const sorted = values
    .filter((value) => Number.isFinite(value) && value > EPSILON)
    .sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function robustMedian(values) {
  const initial = median(values);
  if (initial === null) return null;

  const deviations = values
    .filter((value) => Number.isFinite(value) && value > EPSILON)
    .map((value) => Math.abs(value - initial));
  const mad = median(deviations);
  if (mad === null || mad <= EPSILON) return initial;

  const limit = 3.5 * 1.4826 * mad;
  const filtered = values.filter((value) => Number.isFinite(value) && value > EPSILON && Math.abs(value - initial) <= limit);
  return median(filtered) ?? initial;
}

export function samplePoseSequence(poses, sampleLimit = 360) {
  if (!Array.isArray(poses) || !poses.length) return [];
  if (poses.length <= sampleLimit) return poses;

  const samples = [];
  const lastIndex = poses.length - 1;
  for (let index = 0; index < sampleLimit; index += 1) {
    samples.push(poses[Math.round((index * lastIndex) / (sampleLimit - 1))]);
  }
  return samples;
}

function addDistance(measures, name, pose, first, second) {
  const value = pointDistance(pose[first], pose[second]);
  if (value !== null) measures[name].push(value);
}

function measureOrFallback(measures, name, fallback) {
  return robustMedian(measures[name]) ?? fallback;
}

/**
 * Estimates stable visual radii from a sequence. The limb coefficients mirror
 * MuJoCo Humanoid's radius-to-length ratios rather than frame-local lengths.
 */
export function estimateHuman17Profile(poses) {
  const measures = {
    rightThigh: [], leftThigh: [], rightShin: [], leftShin: [],
    rightUpperArm: [], leftUpperArm: [], rightLowerArm: [], leftLowerArm: [],
    hipWidth: [], shoulderWidth: [], torso: [], neck: [], head: [],
  };

  for (const pose of samplePoseSequence(poses)) {
    if (!isHuman17Pose(pose)) continue;
    addDistance(measures, "rightThigh", pose, 1, 2);
    addDistance(measures, "leftThigh", pose, 4, 5);
    addDistance(measures, "rightShin", pose, 2, 3);
    addDistance(measures, "leftShin", pose, 5, 6);
    addDistance(measures, "rightUpperArm", pose, 14, 15);
    addDistance(measures, "leftUpperArm", pose, 11, 12);
    addDistance(measures, "rightLowerArm", pose, 15, 16);
    addDistance(measures, "leftLowerArm", pose, 12, 13);
    addDistance(measures, "hipWidth", pose, 1, 4);
    addDistance(measures, "shoulderWidth", pose, 11, 14);
    addDistance(measures, "torso", pose, 0, 8);
    addDistance(measures, "neck", pose, 8, 9);
    addDistance(measures, "head", pose, 9, 10);
  }

  const limbLengths = [
    ...measures.rightThigh, ...measures.leftThigh,
    ...measures.rightShin, ...measures.leftShin,
    ...measures.rightUpperArm, ...measures.leftUpperArm,
    ...measures.rightLowerArm, ...measures.leftLowerArm,
  ];
  const referenceLength = robustMedian(limbLengths) ?? 1;
  const rightThigh = measureOrFallback(measures, "rightThigh", referenceLength);
  const leftThigh = measureOrFallback(measures, "leftThigh", referenceLength);
  const rightShin = measureOrFallback(measures, "rightShin", referenceLength);
  const leftShin = measureOrFallback(measures, "leftShin", referenceLength);
  const rightUpperArm = measureOrFallback(measures, "rightUpperArm", referenceLength);
  const leftUpperArm = measureOrFallback(measures, "leftUpperArm", referenceLength);
  const rightLowerArm = measureOrFallback(measures, "rightLowerArm", referenceLength);
  const leftLowerArm = measureOrFallback(measures, "leftLowerArm", referenceLength);
  const hipWidth = measureOrFallback(measures, "hipWidth", referenceLength * 0.58);
  const shoulderWidth = measureOrFallback(measures, "shoulderWidth", referenceLength * 0.76);
  const torsoLength = measureOrFallback(measures, "torso", referenceLength * 1.15);
  const neckLength = measureOrFallback(measures, "neck", referenceLength * 0.22);
  const headLength = measureOrFallback(measures, "head", referenceLength * 0.30);

  return Object.freeze({
    referenceLength,
    lengths: Object.freeze({
      rightThigh, leftThigh, rightShin, leftShin,
      rightUpperArm, leftUpperArm, rightLowerArm, leftLowerArm,
      hipWidth, shoulderWidth, torsoLength, neckLength, headLength,
    }),
    radii: Object.freeze({
      rightThigh: rightThigh * 0.176,
      leftThigh: leftThigh * 0.176,
      rightShin: rightShin * 0.163,
      leftShin: leftShin * 0.163,
      rightUpperArm: rightUpperArm * 0.144,
      leftUpperArm: leftUpperArm * 0.144,
      rightLowerArm: rightLowerArm * 0.112,
      leftLowerArm: leftLowerArm * 0.112,
      pelvis: hipWidth * 0.42,
      shoulders: shoulderWidth * 0.22,
      torsoBottom: hipWidth * 0.34,
      torsoTop: shoulderWidth * 0.24,
      neck: Math.max(neckLength * 0.32, referenceLength * 0.075),
      head: Math.max(headLength * 0.88, referenceLength * 0.27),
      hand: Math.max((rightLowerArm + leftLowerArm) * 0.075, referenceLength * 0.06),
      foot: Math.max((rightShin + leftShin) * 0.05, referenceLength * 0.055),
      joint: referenceLength * 0.075,
    }),
  });
}
