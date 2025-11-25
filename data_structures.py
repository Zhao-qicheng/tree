"""
基础数据结构与工具函数。

包含关键点表示、包围盒运算、八分体判定等通用逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence, Tuple, TypeVar, Union

import numpy as np

import config


Vector3 = np.ndarray
MultiPointOctant = Tuple[int, ...]
T = TypeVar("T")


def _ensure_vector(values: Sequence[float]) -> Vector3:
    """将序列转换为 float64 3 维向量。"""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"期望长度为3的向量，实际形状为 {array.shape}")
    return array


@dataclass
class BodyKeypoints:
    """
    关键点坐标集合（相对于 hip 原点）。

    所有坐标均为 3D numpy 向量，并确保 hip 恒为 [0, 0, 0]。
    字段名使用BVH原始关节名称（10个关键关节）。
    """

    hip: Vector3
    chest: Vector3
    neck: Vector3
    head: Vector3
    lShldr: Vector3
    rShldr: Vector3
    lHand: Vector3
    rHand: Vector3
    lFoot: Vector3
    rFoot: Vector3

    def as_dict(self) -> Dict[str, Vector3]:
        """以字典形式返回关键点，保持名称顺序与 config.KEYPOINT_NAMES 一致。"""
        return {
            name: getattr(self, name)
            for name in config.KEYPOINT_NAMES
        }

    def __iter__(self) -> Iterator[Tuple[str, Vector3]]:
        """按固定顺序迭代 (名称, 坐标)。"""
        for name in config.KEYPOINT_NAMES:
            yield name, getattr(self, name)


@dataclass
class FrameMetadata:
    """
    单个帧的元数据，用于帧检索系统。
    
    存储帧的完整信息，包括原始文件、帧索引和关键点坐标。
    """
    bvh_file: str  # BVH文件路径
    frame_index: int  # 帧索引
    frame_id: str  # 唯一标识，格式如 "01_01_frame_0042"
    keypoints: Dict[str, Vector3]  # 关键点坐标（6位小数精度）
    
    def __post_init__(self):
        """确保关键点坐标精度为6位小数。"""
        for name, vec in self.keypoints.items():
            self.keypoints[name] = np.round(vec, config.JSON_FLOAT_PRECISION)


@dataclass(frozen=True)
class BoundingBox:
    """
    轴对齐包围盒。

    Attributes:
        min_point: 最小角点坐标
        max_point: 最大角点坐标
    """

    min_point: Vector3
    max_point: Vector3

    def center(self) -> Vector3:
        return (self.min_point + self.max_point) / 2.0

    def size(self) -> Vector3:
        return self.max_point - self.min_point

    def subdivide(self, octant: int) -> "BoundingBox":
        """
        根据 octant（0-7）划分子包围盒。

        位意义：bit0->x, bit1->y, bit2->z。1 表示取较大半区。
        """
        if not 0 <= octant <= 7:
            raise ValueError(f"octant 应位于 [0,7]，当前为 {octant}")
        center = self.center()
        min_point = self.min_point.copy()
        max_point = self.max_point.copy()

        if octant & 1:
            min_point[0] = center[0]
        else:
            max_point[0] = center[0]

        if octant & 2:
            min_point[1] = center[1]
        else:
            max_point[1] = center[1]

        if octant & 4:
            min_point[2] = center[2]
        else:
            max_point[2] = center[2]

        return BoundingBox(min_point, max_point)

    def to_tuple(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """用于 JSON 序列化。"""
        return tuple(self.min_point.tolist()), tuple(self.max_point.tolist())

    @staticmethod
    def from_tuple(
        values: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> "BoundingBox":
        """从元组重建 BoundingBox。"""
        min_point, max_point = values
        return BoundingBox(_ensure_vector(min_point), _ensure_vector(max_point))


def normalize_to_hip(raw_keypoints: Mapping[str, Sequence[float]]) -> BodyKeypoints:
    """
    将原始关键点转换为以 hip 为原点的 BodyKeypoints。

    Args:
        raw_keypoints: 关键点名称 -> 绝对坐标序列（包含 hip）
    """
    hip_name = config.KEYPOINT_NAMES[0]  # 第一个关键点应该是 hip
    try:
        hip_vector = _ensure_vector(raw_keypoints[hip_name])
    except KeyError as exc:
        raise KeyError(f"缺失关键点 {hip_name}，无法建立相对坐标系") from exc

    normalized: Dict[str, Vector3] = {}
    for name in config.KEYPOINT_NAMES:
        if name not in raw_keypoints:
            raise KeyError(f"缺失关键点 {name}")
        vec = _ensure_vector(raw_keypoints[name]) - hip_vector
        # 应用精度控制
        normalized[name] = np.round(vec, config.JSON_FLOAT_PRECISION)

    normalized[hip_name] = np.zeros(3, dtype=np.float64)

    # 动态创建 BodyKeypoints，使用配置中的关键点名称
    return BodyKeypoints(**{name: normalized[name] for name in config.KEYPOINT_NAMES})


def compute_octant(point: Vector3, bbox: BoundingBox) -> int:
    """
    根据点和当前包围盒计算 octant 编码。

    若点位于边界，约定落入较大半区（>= center）。
    """
    center = bbox.center()
    octant = 0
    if point[0] >= center[0]:
        octant |= 1
    if point[1] >= center[1]:
        octant |= 2
    if point[2] >= center[2]:
        octant |= 4
    return octant


def compute_combination_index(octants: MultiPointOctant) -> str:
    """
    将多个octant值（0-7）直接拼接成固定长度的字符串索引。
    
    注意：Hips作为原点不参与八叉树迭代，因此octants数量为6个（排除Hips后的关键点）。
    例如：(2, 7, 3, 0, 4, 1) → "273041"
          对应：(LeftHand, RightHand, Neck, LeftFoot, RightFoot, LowerBack)
    """
    return ''.join(str(octant) for octant in octants)


def iterate_keypoints(mapping: MutableMapping[str, T]) -> Iterator[Tuple[str, T]]:
    """
    按 KEYPOINT_NAMES 顺序遍历任意关键点映射。

    有助于保持编码一致性。
    """
    for name in config.KEYPOINT_NAMES:
        yield name, mapping[name]


KeypointInput = Union[BodyKeypoints, Mapping[str, Sequence[float]], Mapping[str, np.ndarray]]


def coerce_body_keypoints(keypoints: KeypointInput) -> BodyKeypoints:
    """
    将输入转换为 BodyKeypoints。

    支持三种形式：
        1. BodyKeypoints（直接返回）
        2. 名称 -> numpy.ndarray
        3. 名称 -> 可转换为 numpy 的序列
    """
    if isinstance(keypoints, BodyKeypoints):
        return keypoints

    converted: Dict[str, Vector3] = {}
    for name in config.KEYPOINT_NAMES:
        if name not in keypoints:
            raise KeyError(f"缺失关键点 {name}")
        value = keypoints[name]
        if isinstance(value, np.ndarray):
            converted[name] = value.astype(np.float64, copy=False)
        else:
            converted[name] = _ensure_vector(value)

    return normalize_to_hip(converted)

