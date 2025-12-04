"""
系统级配置项。

该模块集中定义人体动作八叉树系统所需的常量、关键点列表、空间参数以及推理回退策略。
"""

from __future__ import annotations

from typing import Dict, Tuple

# 关键点名称，使用BVH文件中的43个关键关节
# 这些关节将用于帧检索系统
KEYPOINT_NAMES = (
    "hip",
    "abdomen",
    "chest",
    "neck",
    "head",
    "leftEye",
    "rightEye",
    "rCollar",
    "rShldr",
    "rForeArm",
    "rHand",
    "rThumb1",
    "rThumb2",
    "rIndex1",
    "rIndex2",
    "rMid1",
    "rMid2",
    "rRing1",
    "rRing2",
    "rPinky1",
    "rPinky2",
    "lCollar",
    "lShldr",
    "lForeArm",
    "lHand",
    "lThumb1",
    "lThumb2",
    "lIndex1",
    "lIndex2",
    "lMid1",
    "lMid2",
    "lRing1",
    "lRing2",
    "lPinky1",
    "lPinky2",
    "rButtock",
    "rThigh",
    "rShin",
    "rFoot",
    "lButtock",
    "lThigh",
    "lShin",
    "lFoot",
)

# 用于八叉树迭代的关键点（排除原点 hip）
OCTREE_KEYPOINT_NAMES = tuple(k for k in KEYPOINT_NAMES if k != "hip")

# 八叉树相关配置
MAX_DEPTH: int = 11

# 根节点包围盒尺寸参数（单位：文件中的单位，）
ROOT_BBOX_SIZE: float = 80.0  # 边长
ROOT_BBOX_HALF_SIZE: float = ROOT_BBOX_SIZE / 2.0

# 每个关键点在根节点的初始包围盒范围（min, max）
# 采用相对于 Hips 原点的标准对称立方体 [-80, 80]^3。
ROOT_BOUNDING_BOXES: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    name: (
        (-ROOT_BBOX_HALF_SIZE, -ROOT_BBOX_HALF_SIZE, -ROOT_BBOX_HALF_SIZE),
        (ROOT_BBOX_HALF_SIZE, ROOT_BBOX_HALF_SIZE, ROOT_BBOX_HALF_SIZE),
    )
    for name in KEYPOINT_NAMES
}

# 推理回退策略：当路径中断时，返回最近祖先节点标签。
FALLBACK_STRATEGY: str = "nearest_ancestor"

# 自定义动作标签映射，可根据需要填充。
# 形式：{"custom_label": "说明"}，用于文档化或校验。
ACTION_LABELS: Dict[str, str] = {}

# JSON 持久化设置
JSON_FLOAT_PRECISION: int = 6
JSON_INDENT: int = 2

# 查询配置
TOP_K: int = 5  # 返回最相似的K个帧
EXACT_MATCH_EPSILON: float = 0.01  # 精确匹配的距离阈值
MIN_CANDIDATES: int = 30  # 八叉树查询时的最小候选帧数量（约为总数的1%）
BEAM_WIDTH: int = 4  # 多分枝向下搜索时保留的候选节点数量
MAX_BACKTRACK_DEPTH: int = 2  # 候选不足时允许回溯的最大层级

# 关节对配置：每两个关键点构建一棵八叉树
JOINT_PAIR_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("chest", "neck"),
    ("head", "lShldr"),
    ("head", "rShldr"),
    ("lForeArm", "lHand"),
    ("rForeArm", "rHand"),
    ("lThigh", "lShin"),
    ("rThigh", "rShin"),
    ("lFoot", "rFoot"),
    # 补充一些可能的关节对，可以根据实际需求调整
    ("lCollar", "rCollar"),
    ("abdomen", "chest"),
    ("leftEye", "rightEye"),
    ("rButtock", "lButtock"),
)

# 相似度计算权重（可根据关节重要性调整）
# 默认为1.0，可以根据需要进行微调
JOINT_WEIGHTS: Dict[str, float] = {name: 1.0 for name in KEYPOINT_NAMES}

# 坐标系旋转增强配置
# 通过构建多个旋转坐标系的树来解决边界附近的点被分割的问题
# 建议值: [0, 30, 60] 或 [0, 45]
ROTATION_ANGLES: Tuple[int, ...] = (0,)

# 对特定关节进行权重调整
_CUSTOM_WEIGHTS = {
    "neck": 0.8,
    "head": 0.6,
    "lShldr": 0.8,
    "lForeArm": 1.0,
    "lHand": 1.2,
    "rShldr": 0.8,
    "rForeArm": 1.0,
    "rHand": 1.2,
    "lThigh": 1.1,
    "lShin": 1.1,
    "lFoot": 1.2,
    "rThigh": 1.1,
    "rShin": 1.1,
    "rFoot": 1.2,
    # 手指部分的权重可以稍微降低，或者保持默认
    "lThumb1": 0.8, "lThumb2": 0.8,
    "rThumb1": 0.8, "rThumb2": 0.8,
}
JOINT_WEIGHTS.update(_CUSTOM_WEIGHTS)
