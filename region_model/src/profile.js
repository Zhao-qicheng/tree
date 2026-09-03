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

export const EXTENDED_JOINTS = Object.freeze([
  ...HUMAN17_JOINTS,
  "left_big_toe",
  "left_small_toe",
  "left_heel",
  "right_big_toe",
  "right_small_toe",
  "right_heel",
  "left_thumb_tip",
  "left_index_tip",
  "left_middle_tip",
  "left_ring_tip",
  "left_pinky_tip",
  "right_thumb_tip",
  "right_index_tip",
  "right_middle_tip",
  "right_ring_tip",
  "right_pinky_tip",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "nose_tip",
  "mouth_center",
]);

export const EXTENDED_PARENTS = Object.freeze([
  -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15,
  6, 6, 6, 3, 3, 3,
  13, 13, 13, 13, 13,
  16, 16, 16, 16, 16,
  9, 9, 9, 9, 9, 9,
]);

export const EXTENDED_BONES = Object.freeze(
  EXTENDED_PARENTS
    .map((parent, child) => [parent, child])
    .filter(([parent]) => parent >= 0),
);

export const SELECTED_EXTRA_INDICES = Object.freeze(
  Array.from({ length: 22 }, (_, index) => index + 17),
);

export const SELECTED_EXTRA_BONES = Object.freeze([
  [6, 17], [17, 18], [6, 19],
  [3, 20], [20, 21], [3, 22],
  [13, 23], [13, 24], [13, 25], [13, 26], [13, 27],
  [16, 28], [16, 29], [16, 30], [16, 31], [16, 32],
  [35, 33], [33, 37], [37, 34], [34, 36],
  [33, 34], [37, 38],
]);

function sequentialPairs(start, end, close = false) {
  const pairs = [];
  for (let index = start; index < end; index += 1) {
    pairs.push([index, index + 1]);
  }
  if (close && end > start) pairs.push([end, start]);
  return pairs;
}

function handPairs(offset, bodyWrist) {
  const pairs = [[bodyWrist, offset]];
  for (const start of [1, 5, 9, 13, 17]) {
    pairs.push([offset, offset + start]);
    pairs.push(...sequentialPairs(offset + start, offset + start + 3));
  }
  return pairs;
}

const faceOffset = 23;
export const WHOLEBODY133_BONES = Object.freeze([
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
  [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
  [15, 17], [17, 18], [15, 19],
  [16, 20], [20, 21], [16, 22],
  ...sequentialPairs(faceOffset, faceOffset + 16),
  ...sequentialPairs(faceOffset + 17, faceOffset + 21),
  ...sequentialPairs(faceOffset + 22, faceOffset + 26),
  ...sequentialPairs(faceOffset + 27, faceOffset + 30),
  ...sequentialPairs(faceOffset + 31, faceOffset + 35),
  ...sequentialPairs(faceOffset + 36, faceOffset + 41, true),
  ...sequentialPairs(faceOffset + 42, faceOffset + 47, true),
  ...sequentialPairs(faceOffset + 48, faceOffset + 59, true),
  ...sequentialPairs(faceOffset + 60, faceOffset + 67, true),
  ...handPairs(91, 9),
  ...handPairs(112, 10),
]);

export function isExtendedPose(value) {
  return Array.isArray(value) && value.length === EXTENDED_JOINTS.length;
}

export function corePoseFrom(value) {
  if (isExtendedPose(value)) return value.slice(0, HUMAN17_JOINTS.length);
  return value;
}

export function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vecAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vecScale(a, scale) {
  return [a[0] * scale, a[1] * scale, a[2] * scale];
}

export function vecDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vecCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function vecLength(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vecNormalize(a) {
  if (!a) return null;
  const length = vecLength(a);
  if (!Number.isFinite(length) || length <= EPSILON) return null;
  return vecScale(a, 1 / length);
}

export function vecLerp(a, b, t) {
  return vecAdd(vecScale(a, 1 - t), vecScale(b, t));
}

export function vecMean(points) {
  if (!points.length) return null;
  const sum = points.reduce((acc, point) => vecAdd(acc, point), [0, 0, 0]);
  return vecScale(sum, 1 / points.length);
}

export function posePoint(pose, index, valid) {
  if (valid && valid[index] === false) return null;
  const value = pose?.[index];
  return isFinitePoint(value) ? [Number(value[0]), Number(value[1]), Number(value[2])] : null;
}

export function makeFrame(rightHint, upHint, forwardHint) {
  let right = vecNormalize(rightHint);
  let up = vecNormalize(upHint);
  let forward = vecNormalize(forwardHint);
  if (right && up) {
    up = vecNormalize(vecSub(up, vecScale(right, vecDot(up, right))));
    if (!up) return null;
    forward = vecNormalize(vecCross(right, up));
    if (!forward) return null;
    up = vecNormalize(vecCross(forward, right));
  } else if (right && forward) {
    forward = vecNormalize(vecSub(forward, vecScale(right, vecDot(forward, right))));
    if (!forward) return null;
    up = vecNormalize(vecCross(forward, right));
    if (!up) return null;
    forward = vecNormalize(vecCross(right, up));
  } else if (up && forward) {
    forward = vecNormalize(vecSub(forward, vecScale(up, vecDot(forward, up))));
    if (!forward) return null;
    right = vecNormalize(vecCross(up, forward));
    if (!right) return null;
    up = vecNormalize(vecCross(forward, right));
    forward = vecNormalize(vecCross(right, up));
  } else {
    return null;
  }
  if (!right || !up || !forward) return null;
  return { right, up, forward };
}

export function computeTorsoFrame(pose, valid) {
  const root = posePoint(pose, 0, valid);
  const thorax = posePoint(pose, 8, valid);
  const leftShoulder = posePoint(pose, 11, valid);
  const rightShoulder = posePoint(pose, 14, valid);
  if (!root || !thorax || !leftShoulder || !rightShoulder) return null;
  const frame = makeFrame(vecSub(rightShoulder, leftShoulder), vecSub(thorax, root), null);
  if (!frame) return null;
  return {
    origin: vecLerp(root, thorax, 0.58),
    ...frame,
  };
}

export function computeHeadFrame(pose, valid, torsoFrame = null) {
  const leftEye = posePoint(pose, 33, valid);
  const rightEye = posePoint(pose, 34, valid);
  const leftEar = posePoint(pose, 35, valid);
  const rightEar = posePoint(pose, 36, valid);
  const noseTip = posePoint(pose, 37, valid);
  const mouth = posePoint(pose, 38, valid);
  const nose = posePoint(pose, 9, valid);
  const head = posePoint(pose, 10, valid);
  const thorax = posePoint(pose, 8, valid);

  const rightHint = (leftEye && rightEye)
    ? vecSub(rightEye, leftEye)
    : (leftEar && rightEar)
      ? vecSub(rightEar, leftEar)
      : torsoFrame?.right;
  const upHint = (head && nose)
    ? vecSub(head, nose)
    : (leftEye && rightEye && mouth)
      ? vecSub(vecMean([leftEye, rightEye]), mouth)
      : torsoFrame?.up;
  const earMid = (leftEar && rightEar) ? vecMean([leftEar, rightEar]) : null;
  const eyeMid = (leftEye && rightEye) ? vecMean([leftEye, rightEye]) : null;
  const forwardHint = (noseTip && earMid)
    ? vecSub(noseTip, earMid)
    : (noseTip && eyeMid)
      ? vecSub(noseTip, eyeMid)
      : (nose && thorax)
        ? vecSub(nose, thorax)
        : torsoFrame?.forward;
  const frame = makeFrame(rightHint, upHint, forwardHint);
  if (!frame) return null;
  const origin = noseTip || (nose && head ? vecLerp(nose, head, 0.55) : nose || head || eyeMid);
  if (!origin) return null;
  return { origin, ...frame };
}

export function computeFootFrame(pose, valid, side, torsoFrame = null) {
  const isLeft = side === "left";
  const ankle = posePoint(pose, isLeft ? 6 : 3, valid);
  const bigToe = posePoint(pose, isLeft ? 17 : 20, valid);
  const smallToe = posePoint(pose, isLeft ? 18 : 21, valid);
  const heel = posePoint(pose, isLeft ? 19 : 22, valid);
  if (!ankle) return null;
  const toes = [bigToe, smallToe].filter(Boolean);
  const toeCenter = vecMean(toes);
  if (!toeCenter && !heel) return null;
  const forwardHint = (toeCenter && heel)
    ? vecSub(toeCenter, heel)
    : (toeCenter && ankle)
      ? vecSub(toeCenter, ankle)
      : torsoFrame?.forward;
  const upHint = torsoFrame?.up || [0, 0, 1];
  const frame = makeFrame(null, upHint, forwardHint);
  if (!frame) return null;
  const back = heel || ankle;
  const front = toeCenter || ankle;
  const length = Math.max(vecLength(vecSub(front, back)), EPSILON);
  return {
    origin: vecLerp(back, front, 0.52),
    length,
    width: smallToe && bigToe ? Math.max(vecLength(vecSub(smallToe, bigToe)), length * 0.28) : length * 0.38,
    ...frame,
  };
}

export function computeHandFrame(pose, valid, side, torsoFrame = null) {
  const isLeft = side === "left";
  const wrist = posePoint(pose, isLeft ? 13 : 16, valid);
  const tipStart = isLeft ? 23 : 28;
  const tips = [];
  for (let index = tipStart; index < tipStart + 5; index += 1) {
    const tip = posePoint(pose, index, valid);
    if (tip) tips.push({ index, point: tip });
  }
  if (!wrist || !tips.length) return null;
  const tipCenter = vecMean(tips.map((item) => item.point));
  const forwardHint = vecSub(tipCenter, wrist);
  const thumb = posePoint(pose, tipStart, valid);
  const pinky = posePoint(pose, tipStart + 4, valid);
  const rightHint = (thumb && pinky)
    ? (isLeft ? vecSub(thumb, pinky) : vecSub(pinky, thumb))
    : torsoFrame?.right;
  const frame = makeFrame(rightHint, torsoFrame?.up || [0, 0, 1], forwardHint);
  if (!frame) return null;
  return {
    origin: vecLerp(wrist, tipCenter, 0.35),
    length: Math.max(vecLength(forwardHint), EPSILON),
    tips,
    wrist,
    ...frame,
  };
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
    const corePose = corePoseFrom(pose);
    if (!isHuman17Pose(corePose)) continue;
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
