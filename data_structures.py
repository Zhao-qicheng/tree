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
    字段名使用BVH原始关节名称。
    """
    # 使用 __dict__ 动态存储属性，或者显式列出所有属性
    # 为了保持代码清晰和类型提示，我们显式列出所有43个关节
    hip: Vector3
    abdomen: Vector3
    chest: Vector3
    neck: Vector3
    head: Vector3
    leftEye: Vector3
    rightEye: Vector3
    rCollar: Vector3
    rShldr: Vector3
    rForeArm: Vector3
    rHand: Vector3
    rThumb1: Vector3
    rThumb2: Vector3
    rIndex1: Vector3
    rIndex2: Vector3
    rMid1: Vector3
    rMid2: Vector3
    rRing1: Vector3
    rRing2: Vector3
    rPinky1: Vector3
    rPinky2: Vector3
    lCollar: Vector3
    lShldr: Vector3
    lForeArm: Vector3
    lHand: Vector3
    lThumb1: Vector3
    lThumb2: Vector3
    lIndex1: Vector3
    lIndex2: Vector3
    lMid1: Vector3
    lMid2: Vector3
    lRing1: Vector3
    lRing2: Vector3
    lPinky1: Vector3
    lPinky2: Vector3
    rButtock: Vector3
    rThigh: Vector3
    rShin: Vector3
    rFoot: Vector3
    lButtock: Vector3
    lThigh: Vector3
    lShin: Vector3
    lFoot: Vector3

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


class FrameMetadata:
    """
    单个帧的元数据，用于帧检索系统。
    
    存储帧的完整信息，包括原始文件、帧索引和关键点坐标。
    内部使用紧凑的numpy数组存储关键点以节省内存。
    """
    __slots__ = ("bvh_file", "frame_index", "frame_id", "_keypoints_array")

    def __init__(self, bvh_file: str, frame_index: int, frame_id: str, keypoints: Union[Dict[str, Vector3], np.ndarray]):
        self.bvh_file = bvh_file
        self.frame_index = frame_index
        self.frame_id = frame_id
        
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim != 2 or keypoints.shape[1] != 3:
                # 尝试重塑
                if keypoints.size % 3 == 0:
                    keypoints = keypoints.reshape(-1, 3)
                else:
                    raise ValueError(f"keypoints array must be shape (N, 3), got {keypoints.shape}")
            self._keypoints_array = keypoints.astype(np.float32)
        else:
            # Convert dict to array
            num_joints = len(config.KEYPOINT_NAMES)
            self._keypoints_array = np.zeros((num_joints, 3), dtype=np.float32)
            for i, name in enumerate(config.KEYPOINT_NAMES):
                if name in keypoints:
                    vec = keypoints[name]
                    self._keypoints_array[i] = np.round(vec, config.JSON_FLOAT_PRECISION).astype(np.float32)
                else:
                    # 如果缺失，填0或处理
                    pass

    @property
    def keypoints(self) -> Dict[str, Vector3]:
        """返回关键点字典，保持向后兼容。"""
        kps = {}
        for i, name in enumerate(config.KEYPOINT_NAMES):
             if i < len(self._keypoints_array):
                 kps[name] = self._keypoints_array[i].astype(np.float64)
        return kps
        
    def get_keypoints_array(self) -> np.ndarray:
        """获取原始关键点数组 (float32)"""
        return self._keypoints_array

    def __repr__(self):
        return f"FrameMetadata(bvh_file='{self.bvh_file}', frame_index={self.frame_index}, frame_id='{self.frame_id}')"


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
            # 为了兼容性，如果某些关节缺失（例如不同BVH文件结构略有差异），
            # 可以选择填充0或者抛出异常。这里严格要求所有配置的关键点都存在。
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


def rotate_point_y(point: Vector3, angle_degrees: float) -> Vector3:
    """
    绕Y轴旋转点。
    
    Args:
        point: [x, y, z] 向量
        angle_degrees: 旋转角度（度）
    
    Returns:
        旋转后的新向量
    """
    if angle_degrees == 0:
        return point
    
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    
    x, y, z = point
    # 顺时针/逆时针取决于坐标系定义，这里使用标准旋转矩阵：
    # x' = x*cos - z*sin
    # z' = x*sin + z*cos
    # 注意：BVH通常是Y轴向上。
    new_x = x * c - z * s
    new_z = x * s + z * c
    
    return np.array([new_x, y, new_z], dtype=point.dtype)


def rotate_keypoints(keypoints: Dict[str, Vector3], angle_degrees: float) -> Dict[str, Vector3]:
    """
    旋转所有关键点。
    """
    if angle_degrees == 0:
        return keypoints.copy()
        
    return {
        k: rotate_point_y(v, angle_degrees)
        for k, v in keypoints.items()
    }
