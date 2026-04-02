"""
姿态模板聚类所需的三层特征提取。

主流程：
1. 对单帧进行 hip 中心化、方向归一化、标准骨架归一化。
2. 提取三层特征：
   - 第一层：关节角度
   - 第二层：姿态结构特征
   - 第三层：花样滑冰关键特征
3. 输出：
   - 原始特征（便于解释与规则生成）
   - 数值特征向量（便于标准化、PCA 与聚类）
   - 类别特征（便于模板统计）
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np

import config
from data_structures import BoundingBox, compute_octant
from npy_loader import H36M_JOINT_NAMES, align_orientation, normalize_skeleton

Vector3 = np.ndarray
KeypointInput = Union[Mapping[str, Sequence[float]], np.ndarray]
VERTICAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class PoseFeatureBundle:
    raw_features: Dict[str, float | str]
    report_features: Dict[str, float]
    categorical_features: Dict[str, str]
    vector_features: Dict[str, float]
    vector: np.ndarray
    vector_feature_names: List[str]
    feature_source: str = "handcrafted"


def _as_vector(values: Sequence[float]) -> Vector3:
    vec = np.asarray(values, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"关键点坐标必须是长度为3的向量，实际为 {vec.shape}")
    return vec


def keypoints_to_dict(keypoints: KeypointInput) -> Dict[str, Vector3]:
    if isinstance(keypoints, np.ndarray):
        if keypoints.shape != (len(H36M_JOINT_NAMES), 3):
            raise ValueError(
                f"当输入为 numpy 数组时，形状必须为 ({len(H36M_JOINT_NAMES)}, 3)，实际为 {keypoints.shape}"
            )
        return {name: keypoints[idx].astype(np.float64, copy=False) for idx, name in enumerate(H36M_JOINT_NAMES)}

    converted: Dict[str, Vector3] = {}
    for name in config.KEYPOINT_NAMES:
        if name not in keypoints:
            raise KeyError(f"缺失关键点: {name}")
        converted[name] = _as_vector(keypoints[name])
    return converted


def preprocess_frame(frame: np.ndarray, *, align: bool = True, normalize: bool = True) -> np.ndarray:
    """统一的单帧预处理：中心化 -> 朝向归一化 -> 尺度归一化。"""
    processed = np.asarray(frame, dtype=np.float64).copy()
    if processed.shape != (len(H36M_JOINT_NAMES), 3):
        raise ValueError(f"单帧数据形状必须为 ({len(H36M_JOINT_NAMES)}, 3)，实际为 {processed.shape}")

    if align:
        processed = align_orientation(processed)
    else:
        processed = processed - processed[0]

    if normalize:
        processed = normalize_skeleton(processed)
    return processed


def _safe_norm(vec: Vector3, eps: float = 1e-8) -> float:
    value = float(np.linalg.norm(vec))
    return value if value > eps else eps


def _angle_between(v1: Vector3, v2: Vector3) -> float:
    denom = _safe_norm(v1) * _safe_norm(v2)
    cosine = float(np.dot(v1, v2) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _angle_degrees(a: Vector3, b: Vector3, c: Vector3) -> float:
    return _angle_between(a - b, c - b)


def _planar_angle(vec: Vector3) -> float:
    """返回向量在 XY 平面的角度（度）。"""
    if _safe_norm(vec[:2]) <= 1e-8:
        return 0.0
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def _resolve_support_leg(kp: Dict[str, Vector3], left_knee: float, right_knee: float) -> str:
    """
    使用“更直的腿 + 更低的踝”联合判断支撑腿。
    """
    left_score = left_knee - 0.5 * kp["lAnkle"][2]
    right_score = right_knee - 0.5 * kp["rAnkle"][2]
    delta = left_score - right_score
    if delta > 5.0:
        return "left"
    if delta < -5.0:
        return "right"
    return "unknown"


def _torso_reference(kp: Dict[str, Vector3]) -> Vector3:
    return (kp["neck"] - kp["hip"]).astype(np.float64, copy=False)


def _build_bundle(
    *,
    raw_features: Dict[str, float | str],
    report_features: Dict[str, float],
    categorical_features: Dict[str, str],
    vector_features: Dict[str, float],
    feature_source: str,
) -> PoseFeatureBundle:
    vector_feature_names = list(vector_features.keys())
    vector = np.array([vector_features[name] for name in vector_feature_names], dtype=np.float64)
    return PoseFeatureBundle(
        raw_features=raw_features,
        report_features=report_features,
        categorical_features=categorical_features,
        vector_features=vector_features,
        vector=vector,
        vector_feature_names=vector_feature_names,
        feature_source=feature_source,
    )


def _root_bbox_for_joint(name: str) -> BoundingBox:
    bounds = config.ROOT_BOUNDING_BOXES.get(name)
    if bounds is None:
        raise KeyError(f"缺少关节 {name} 的根包围盒配置")
    return BoundingBox.from_tuple(bounds)


def _joint_octant_path(point: Vector3, depth: int, bbox: BoundingBox) -> list[int]:
    current_bbox = bbox
    octants: list[int] = []
    for _ in range(depth):
        octant = compute_octant(point, current_bbox)
        octants.append(octant)
        current_bbox = current_bbox.subdivide(octant)
    return octants


def _octant_path_to_leaf_index(octants: Sequence[int]) -> int:
    index = 0
    for octant in octants:
        index = index * 8 + int(octant)
    return index


def extract_three_level_features(keypoints: KeypointInput) -> PoseFeatureBundle:
    """
    提取三层姿态特征。

    约定：
    - 竖直方向为 z 轴
    - 输入建议已经过对齐和尺度归一化；若没有，也可直接计算
    """
    kp = keypoints_to_dict(keypoints)
    hip = kp["hip"]
    centered = {name: value - hip for name, value in kp.items()}

    torso_vec = _torso_reference(centered)
    torso_length = _safe_norm(torso_vec)
    hip_width = _safe_norm(centered["lHip"] - centered["rHip"])
    shoulder_width = _safe_norm(centered["lShoulder"] - centered["rShoulder"])
    left_leg_length = _safe_norm(centered["lHip"] - centered["lKnee"]) + _safe_norm(centered["lKnee"] - centered["lAnkle"])
    right_leg_length = _safe_norm(centered["rHip"] - centered["rKnee"]) + _safe_norm(centered["rKnee"] - centered["rAnkle"])
    leg_length = max((left_leg_length + right_leg_length) / 2.0, 1e-8)

    knee_left = _angle_degrees(centered["lHip"], centered["lKnee"], centered["lAnkle"])
    knee_right = _angle_degrees(centered["rHip"], centered["rKnee"], centered["rAnkle"])
    hip_left = _angle_between(centered["spine"] - centered["lHip"], centered["lKnee"] - centered["lHip"])
    hip_right = _angle_between(centered["spine"] - centered["rHip"], centered["rKnee"] - centered["rHip"])
    elbow_left = _angle_degrees(centered["lShoulder"], centered["lElbow"], centered["lWrist"])
    elbow_right = _angle_degrees(centered["rShoulder"], centered["rElbow"], centered["rWrist"])
    torso_angle = _angle_between(torso_vec, VERTICAL_AXIS)

    arm_span_ratio = float(np.linalg.norm(centered["lWrist"] - centered["rWrist"])) / shoulder_width
    feet_distance_ratio = float(np.linalg.norm(centered["lAnkle"] - centered["rAnkle"])) / hip_width
    hand_height_ratio = (((centered["lWrist"][2] + centered["rWrist"][2]) / 2.0) - centered["hip"][2]) / torso_length

    support_leg_type = _resolve_support_leg(centered, knee_left, knee_right)
    if support_leg_type == "left":
        support_hip = centered["lHip"]
        support_ankle = centered["lAnkle"]
        free_ankle = centered["rAnkle"]
        support_one_hot = {"support_leg_is_left": 1.0, "support_leg_is_right": 0.0}
    elif support_leg_type == "right":
        support_hip = centered["rHip"]
        support_ankle = centered["rAnkle"]
        free_ankle = centered["lAnkle"]
        support_one_hot = {"support_leg_is_left": 0.0, "support_leg_is_right": 1.0}
    else:
        support_hip = centered["hip"]
        support_ankle = (centered["lAnkle"] + centered["rAnkle"]) / 2.0
        free_ankle = support_ankle
        support_one_hot = {"support_leg_is_left": 0.0, "support_leg_is_right": 0.0}

    free_leg_height = float(free_ankle[2] - support_ankle[2]) / leg_length
    free_leg_vector = free_ankle - support_hip
    free_leg_direction = _planar_angle(free_leg_vector)
    torso_lean_direction = _planar_angle(torso_vec)

    report_features: Dict[str, float] = {
        "knee_left": knee_left,
        "knee_right": knee_right,
        "hip_left": hip_left,
        "hip_right": hip_right,
        "elbow_left": elbow_left,
        "elbow_right": elbow_right,
        "torso_angle": torso_angle,
        "arm_span_ratio": arm_span_ratio,
        "feet_distance_ratio": feet_distance_ratio,
        "hand_height_ratio": hand_height_ratio,
        "free_leg_height": free_leg_height,
        "free_leg_direction": free_leg_direction,
        "torso_lean_direction": torso_lean_direction,
    }

    categorical_features = {"support_leg_type": support_leg_type}

    raw_features: Dict[str, float | str] = dict(report_features)
    raw_features.update(categorical_features)

    vector_features: Dict[str, float] = {
        "knee_left": knee_left,
        "knee_right": knee_right,
        "hip_left": hip_left,
        "hip_right": hip_right,
        "elbow_left": elbow_left,
        "elbow_right": elbow_right,
        "torso_angle": torso_angle,
        "arm_span_ratio": arm_span_ratio,
        "feet_distance_ratio": feet_distance_ratio,
        "hand_height_ratio": hand_height_ratio,
        "free_leg_height": free_leg_height,
        "free_leg_direction_sin": math.sin(math.radians(free_leg_direction)),
        "free_leg_direction_cos": math.cos(math.radians(free_leg_direction)),
        "torso_lean_direction_sin": math.sin(math.radians(torso_lean_direction)),
        "torso_lean_direction_cos": math.cos(math.radians(torso_lean_direction)),
        **support_one_hot,
    }

    return _build_bundle(
        raw_features=raw_features,
        report_features=report_features,
        categorical_features=categorical_features,
        vector_features=vector_features,
        feature_source="handcrafted",
    )


def extract_coordinate_features(
    keypoints: KeypointInput,
    *,
    coordinate_mode: str = "flattened",
) -> PoseFeatureBundle:
    if coordinate_mode != "flattened":
        raise ValueError(f"暂不支持的坐标特征模式: {coordinate_mode}")

    kp = keypoints_to_dict(keypoints)
    raw_features: Dict[str, float | str] = {}
    report_features: Dict[str, float] = {}
    vector_features: Dict[str, float] = {}
    for name in config.KEYPOINT_NAMES:
        vec = kp[name]
        for axis_name, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
            feature_name = f"{name}_{axis_name}"
            value = float(vec[axis_idx])
            raw_features[feature_name] = value
            report_features[feature_name] = value
            vector_features[feature_name] = value

    return _build_bundle(
        raw_features=raw_features,
        report_features=report_features,
        categorical_features={},
        vector_features=vector_features,
        feature_source="coordinates",
    )


def extract_octree_leaf_index_features(
    keypoints: KeypointInput,
    *,
    depth: int | None = None,
) -> PoseFeatureBundle:
    kp = keypoints_to_dict(keypoints)
    feature_depth = depth if depth is not None else max(1, int(config.MAX_DEPTH))

    raw_features: Dict[str, float | str] = {}
    report_features: Dict[str, float] = {}
    vector_features: Dict[str, float] = {}
    for name in config.KEYPOINT_NAMES:
        bbox = _root_bbox_for_joint(name)
        octants = _joint_octant_path(kp[name], feature_depth, bbox)
        leaf_index = float(_octant_path_to_leaf_index(octants))
        feature_name = f"leaf_idx_{name}"
        raw_features[feature_name] = leaf_index
        report_features[feature_name] = leaf_index
        vector_features[feature_name] = leaf_index

    return _build_bundle(
        raw_features=raw_features,
        report_features=report_features,
        categorical_features={},
        vector_features=vector_features,
        feature_source="octree_leaf_index_17d",
    )


def extract_octree_node_features(
    keypoints: KeypointInput,
    *,
    octree_mode: str = "path_encoding",
    depth: int | None = None,
) -> PoseFeatureBundle:
    if octree_mode == "joint_leaf_index_17d":
        return extract_octree_leaf_index_features(keypoints, depth=depth)
    if octree_mode != "path_encoding":
        raise ValueError(f"暂不支持的八叉树特征模式: {octree_mode}")

    kp = keypoints_to_dict(keypoints)
    feature_depth = depth if depth is not None else max(1, int(config.MAX_DEPTH))

    raw_features: Dict[str, float | str] = {}
    report_features: Dict[str, float] = {}
    vector_features: Dict[str, float] = {}
    for name in config.KEYPOINT_NAMES:
        bbox = _root_bbox_for_joint(name)
        octants = _joint_octant_path(kp[name], feature_depth, bbox)
        raw_features[f"path_{name}"] = "-".join(str(o) for o in octants)
        for depth_idx, octant in enumerate(octants):
            feature_name = f"path_{name}_d{depth_idx}"
            value = float(octant)
            report_features[feature_name] = value
            vector_features[feature_name] = value

    return _build_bundle(
        raw_features=raw_features,
        report_features=report_features,
        categorical_features={},
        vector_features=vector_features,
        feature_source="octree_node",
    )


def extract_pose_features_by_source(
    feature_source: str,
    keypoints: KeypointInput,
    *,
    coordinate_mode: str | None = None,
    octree_mode: str | None = None,
) -> PoseFeatureBundle:
    if feature_source == "handcrafted":
        return extract_three_level_features(keypoints)
    if feature_source == "coordinates":
        return extract_coordinate_features(
            keypoints,
            coordinate_mode=coordinate_mode or config.POSE_TEMPLATE_COORDINATE_MODE,
        )
    if feature_source == "octree_node":
        return extract_octree_node_features(
            keypoints,
            octree_mode=octree_mode or config.POSE_TEMPLATE_OCTREE_NODE_MODE,
        )
    raise ValueError(f"未知的姿态模板特征源: {feature_source}")

