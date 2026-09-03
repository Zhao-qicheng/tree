"""H36M-17 + COCO-WholeBody 扩展点协议。

第一版固定 39 点：核心 17 点始终来自 MotionAGFormer，扩展 22 点
（双脚 6、指尖 10、面部方向 6）来自 RTMW3D 133 点筛选与融合。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


SCHEMA_VERSION = "h36m17_extended39_v1"
NUM_CORE_JOINTS = 17
NUM_EXTENDED_JOINTS = 39
NUM_WHOLEBODY_JOINTS = 133

H36M_JOINT_NAMES: Tuple[str, ...] = (
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
)

EXTENDED_JOINT_NAMES: Tuple[str, ...] = H36M_JOINT_NAMES + (
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
)

# child -> parent，root 的父节点为 -1
EXTENDED_PARENTS: Tuple[int, ...] = (
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15,
    6, 6, 6, 3, 3, 3,
    13, 13, 13, 13, 13,
    16, 16, 16, 16, 16,
    9, 9, 9, 9, 9, 9,
)

EXTENDED_BONES: Tuple[Tuple[int, int], ...] = tuple(
    (parent, child)
    for child, parent in enumerate(EXTENDED_PARENTS)
    if parent >= 0
)

# MotionAGFormer/H36M 与 COCO-WholeBody 身体点的共用锚点
SHARED_ANCHORS: Tuple[Tuple[int, int], ...] = (
    (1, 12),   # right_hip
    (2, 14),   # right_knee
    (3, 16),   # right_ankle
    (4, 11),   # left_hip
    (5, 13),   # left_knee
    (6, 15),   # left_ankle
    (9, 0),    # nose
    (11, 5),   # left_shoulder
    (12, 7),   # left_elbow
    (13, 9),   # left_wrist
    (14, 6),   # right_shoulder
    (15, 8),   # right_elbow
    (16, 10),  # right_wrist
)

PELVIS_WHOLEBODY_INDICES: Tuple[int, ...] = (11, 12)

# 扩展点在 39 点协议中的切片
FOOT_INDICES: Tuple[int, ...] = tuple(range(17, 23))
HAND_INDICES: Tuple[int, ...] = tuple(range(23, 33))
FACE_INDICES: Tuple[int, ...] = tuple(range(33, 39))
LEFT_FOOT_INDICES: Tuple[int, ...] = (17, 18, 19)
RIGHT_FOOT_INDICES: Tuple[int, ...] = (20, 21, 22)
LEFT_HAND_INDICES: Tuple[int, ...] = tuple(range(23, 28))
RIGHT_HAND_INDICES: Tuple[int, ...] = tuple(range(28, 33))

LEFT_FOOT_PARENT = 6
RIGHT_FOOT_PARENT = 3
LEFT_HAND_PARENT = 13
RIGHT_HAND_PARENT = 16
FACE_PARENT = 9

ATTACHMENT_GROUPS: Tuple[Dict[str, object], ...] = (
    {"name": "left_foot", "indices": LEFT_FOOT_INDICES, "parent": LEFT_FOOT_PARENT},
    {"name": "right_foot", "indices": RIGHT_FOOT_INDICES, "parent": RIGHT_FOOT_PARENT},
    {"name": "left_hand", "indices": LEFT_HAND_INDICES, "parent": LEFT_HAND_PARENT},
    {"name": "right_hand", "indices": RIGHT_HAND_INDICES, "parent": RIGHT_HAND_PARENT},
    {"name": "face", "indices": FACE_INDICES, "parent": FACE_PARENT},
)

# 68 点面部在 WholeBody 中从 23 开始：
# 右眼 36-41、左眼 42-47、鼻尖 30、上唇 51、下唇 57
WholeBodySource = Union[int, Tuple[int, ...]]

WHOLEBODY_SOURCES: Dict[int, WholeBodySource] = {
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 95,   # left thumb tip  91+4
    24: 99,   # left index tip  91+8
    25: 103,  # left middle tip 91+12
    26: 107,  # left ring tip   91+16
    27: 111,  # left pinky tip  91+20
    28: 116,  # right thumb tip 112+4
    29: 120,
    30: 124,
    31: 128,
    32: 132,
    33: (65, 66, 67, 68, 69, 70),  # left eye
    34: (59, 60, 61, 62, 63, 64),  # right eye
    35: 3,   # left ear
    36: 4,   # right ear
    37: 53,  # nose tip (face landmark 30)
    38: (74, 80),  # mouth center from outer upper/lower lip
}

WHOLEBODY_FALLBACKS: Dict[int, WholeBodySource] = {
    33: 1,  # COCO left_eye
    34: 2,  # COCO right_eye
    37: 0,  # COCO nose
    38: 0,
}

SOURCE_STATUS_CORE = 0
SOURCE_STATUS_EXTENDED = 1
SOURCE_STATUS_INTERPOLATED = 2
SOURCE_STATUS_HIDDEN = 3
SOURCE_STATUS_REJECTED = 4


def joint_index(name: str) -> int:
    try:
        return EXTENDED_JOINT_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown extended joint: {name}") from exc


def is_extended_pose(value: np.ndarray) -> bool:
    array = np.asarray(value)
    return array.ndim >= 2 and array.shape[-2] == NUM_EXTENDED_JOINTS and array.shape[-1] >= 3


def is_core_pose(value: np.ndarray) -> bool:
    array = np.asarray(value)
    return array.ndim >= 2 and array.shape[-2] == NUM_CORE_JOINTS and array.shape[-1] >= 3


def _as_index_list(source: WholeBodySource) -> List[int]:
    if isinstance(source, (int, np.integer)):
        return [int(source)]
    return [int(index) for index in source]


def _gather_source(
    keypoints: np.ndarray,
    scores: np.ndarray,
    source: WholeBodySource,
    score_threshold: float,
) -> Tuple[Optional[np.ndarray], float]:
    indices = _as_index_list(source)
    valid_points = []
    valid_scores = []
    for index in indices:
        if index < 0 or index >= keypoints.shape[0]:
            continue
        point = np.asarray(keypoints[index], dtype=np.float64)
        score = float(scores[index]) if index < len(scores) else 0.0
        if not np.all(np.isfinite(point)) or score < score_threshold:
            continue
        valid_points.append(point)
        valid_scores.append(score)
    if not valid_points:
        return None, 0.0
    return np.mean(np.stack(valid_points, axis=0), axis=0), float(np.mean(valid_scores))


def extract_extended_from_wholebody(
    keypoints_133: np.ndarray,
    scores_133: np.ndarray,
    score_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从单帧或序列的 133 点提取扩展 39 点中的 17–38 号。

    核心 0–16 保持 NaN，由融合阶段写入 MotionAGFormer 结果。
    返回 coords (..., 39, 3)、scores (..., 39)、valid (..., 39)。
    """
    keypoints = np.asarray(keypoints_133, dtype=np.float64)
    scores = np.asarray(scores_133, dtype=np.float64)
    squeeze_batch = False
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
        scores = scores[None]
        squeeze_batch = True
    if keypoints.ndim != 3 or keypoints.shape[1] != NUM_WHOLEBODY_JOINTS or keypoints.shape[2] < 3:
        raise ValueError(f"期望 (T, 133, 3) WholeBody 关键点，实际 {keypoints.shape}")
    if scores.shape[:2] != keypoints.shape[:2]:
        raise ValueError(f"scores 形状不匹配: {scores.shape} vs {keypoints.shape}")

    t_count = keypoints.shape[0]
    coords = np.full((t_count, NUM_EXTENDED_JOINTS, 3), np.nan, dtype=np.float64)
    out_scores = np.zeros((t_count, NUM_EXTENDED_JOINTS), dtype=np.float64)
    valid = np.zeros((t_count, NUM_EXTENDED_JOINTS), dtype=bool)

    for t in range(t_count):
        frame_kpts = keypoints[t, :, :3]
        frame_scores = scores[t]
        for ext_index, source in WHOLEBODY_SOURCES.items():
            point, score = _gather_source(frame_kpts, frame_scores, source, score_threshold)
            if point is None and ext_index in WHOLEBODY_FALLBACKS:
                point, score = _gather_source(
                    frame_kpts, frame_scores, WHOLEBODY_FALLBACKS[ext_index], score_threshold
                )
            if point is None:
                continue
            coords[t, ext_index] = point
            out_scores[t, ext_index] = score
            valid[t, ext_index] = True

    if squeeze_batch:
        return coords[0], out_scores[0], valid[0]
    return coords, out_scores, valid


def shared_anchor_points(
    core_pose: np.ndarray,
    wholebody_pose: np.ndarray,
    wholebody_scores: Optional[np.ndarray] = None,
    score_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取一帧中同时有效的 H36M / WholeBody 共用锚点。"""
    core = np.asarray(core_pose, dtype=np.float64)
    whole = np.asarray(wholebody_pose, dtype=np.float64)
    scores = (
        np.ones((NUM_WHOLEBODY_JOINTS,), dtype=np.float64)
        if wholebody_scores is None
        else np.asarray(wholebody_scores, dtype=np.float64)
    )
    src = []
    dst = []
    names = []
    for core_index, wb_index in SHARED_ANCHORS:
        src_point = whole[wb_index, :3]
        dst_point = core[core_index, :3]
        score = float(scores[wb_index]) if wb_index < len(scores) else 1.0
        if (
            np.all(np.isfinite(src_point))
            and np.all(np.isfinite(dst_point))
            and score >= score_threshold
        ):
            src.append(src_point)
            dst.append(dst_point)
            names.append(core_index)
    if not src:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, np.zeros((0,), dtype=np.int32)
    return np.stack(src, axis=0), np.stack(dst, axis=0), np.asarray(names, dtype=np.int32)


def pelvis_point(wholebody_pose: np.ndarray, wholebody_scores: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    pose = np.asarray(wholebody_pose, dtype=np.float64)
    scores = (
        np.ones((pose.shape[0],), dtype=np.float64)
        if wholebody_scores is None
        else np.asarray(wholebody_scores, dtype=np.float64)
    )
    points = []
    for index in PELVIS_WHOLEBODY_INDICES:
        if index < pose.shape[0] and np.all(np.isfinite(pose[index])) and float(scores[index]) > 0:
            points.append(pose[index, :3])
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def default_extended_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": 60.0,
        "score_threshold": 0.25,
        "shared_anchor_min": 6,
        "residual_threshold": 0.65,
        "scale_window": 31,
        "max_gap": 8,
        "repaired_score": 0.35,
        "length_clip": [0.25, 2.8],
        "one_euro": {
            "enabled": True,
            "min_cutoff": 1.2,
            "beta": 0.01,
            "dcutoff": 1.0,
        },
    }


def schema_metadata() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "joint_names": list(EXTENDED_JOINT_NAMES),
        "parents": list(EXTENDED_PARENTS),
        "bones": [list(pair) for pair in EXTENDED_BONES],
        "num_core_joints": NUM_CORE_JOINTS,
        "num_extended_joints": NUM_EXTENDED_JOINTS,
        "coordinate_space": "root_relative_image_units",
        "core_source": "motionagformer_h36m17",
        "extended_source": "rtmw3d_coco_wholebody133",
    }
