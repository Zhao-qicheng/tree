"""
基础数据结构与工具函数。

包含关键点表示、包围盒运算、八分体判定等通用逻辑。
支持 Human3.6M 17关节格式。
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
    使用 Human3.6M 17关节格式。
    """

    hip: Vector3        # 0 - 髋部（原点）
    rHip: Vector3       # 1 - 右髋
    rKnee: Vector3      # 2 - 右膝
    rAnkle: Vector3     # 3 - 右踝
    lHip: Vector3       # 4 - 左髋
    lKnee: Vector3      # 5 - 左膝
    lAnkle: Vector3     # 6 - 左踝
    spine: Vector3      # 7 - 脊柱
    chest: Vector3      # 8 - 胸部
    neck: Vector3       # 9 - 颈部
    head: Vector3       # 10 - 头部
    lShoulder: Vector3  # 11 - 左肩
    lElbow: Vector3     # 12 - 左肘
    lWrist: Vector3     # 13 - 左腕
    rShoulder: Vector3  # 14 - 右肩
    rElbow: Vector3     # 15 - 右肘
    rWrist: Vector3     # 16 - 右腕

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
    source_file: str  # 源文件路径（BVH 或 NPY）
    frame_index: int  # 帧索引
    frame_id: str  # 唯一标识
    keypoints: Dict[str, Vector3]  # 关键点坐标（6位小数精度）
    
    # 为兼容性保留
    @property
    def bvh_file(self) -> str:
        return self.source_file
    
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
    hip_name = config.KEYPOINT_NAMES[0]  # 第一个关键点是 hip
    try:
        hip_vector = _ensure_vector(raw_keypoints[hip_name])
    except KeyError as exc:
        raise KeyError(f"缺失关键点 {hip_name}，无法建立相对坐标系") from exc

    normalized: Dict[str, Vector3] = {}
    for name in config.KEYPOINT_NAMES:
        if name not in raw_keypoints:
            raise KeyError(f"缺失关键点 {name}")
        vec = _ensure_vector(raw_keypoints[name]) - hip_vector
        normalized[name] = np.round(vec, config.JSON_FLOAT_PRECISION)

    normalized[hip_name] = np.zeros(3, dtype=np.float64)

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
    将多个octant值（0-7）直接拼接成字符串索引。
    """
    return ''.join(str(octant) for octant in octants)


def iterate_keypoints(mapping: MutableMapping[str, T]) -> Iterator[Tuple[str, T]]:
    """
    按 KEYPOINT_NAMES 顺序遍历任意关键点映射。
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
