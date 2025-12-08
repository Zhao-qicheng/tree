"""
系统级配置项。

该模块集中定义人体动作八叉树系统所需的常量、关键点列表、空间参数以及推理回退策略。
"""

from __future__ import annotations

from typing import Dict, Tuple

# 关键点名称，使用BVH文件中的10个关键关节
# 这些关节将用于帧检索系统
KEYPOINT_NAMES = (
    "hip",        # 原点，始终为 [0, 0, 0]
    "chest",      # 胸部
    "neck",       # 颈部
    "head",       # 头部
    "lShldr",     # 左肩
    "rShldr",     # 右肩
    "lHand",      # 左手
    "rHand",      # 右手
    "lFoot",      # 左脚
    "rFoot",      # 右脚
)

# 用于八叉树迭代的关键点（排除原点 hip）
OCTREE_KEYPOINT_NAMES = (
    "chest",
    "neck",
    "head",
    "lShldr",
    "rShldr",
    "lHand",
    "rHand",
    "lFoot",
    "rFoot",
)

# 关节对配置
# 按顺序每两个关节一组
JOINT_PAIRS: list[tuple[str, ...]] = []
_it = iter(OCTREE_KEYPOINT_NAMES)
for _item in _it:
    try:
        JOINT_PAIRS.append((_item, next(_it)))
    except StopIteration:
        JOINT_PAIRS.append((_item,))

# 八叉树相关配置
<<<<<<< Updated upstream
MAX_DEPTH: int = 12
=======
PAIR_DEPTH: int = 2  # n: 每个关节对的迭代深度
MAX_DEPTH: int = len(JOINT_PAIRS) * PAIR_DEPTH
>>>>>>> Stashed changes

# 根节点包围盒尺寸参数（单位：文件中的单位，）
ROOT_BBOX_SIZE: float = 80.0  # 边长
ROOT_BBOX_HALF_SIZE: float = ROOT_BBOX_SIZE / 2.0

# 每个关键点在根节点的初始包围盒范围（min, max）
# 采用相对于 Hips 原点的标准对称立方体 [-100, 100]^3（单位：厘米）。
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
EXACT_MATCH_EPSILON: float = 0.001  # 精确匹配的距离阈值

# 相似度计算权重（可根据关节重要性调整）
JOINT_WEIGHTS: Dict[str, float] = {
    "hip": 1.0,
    "chest": 1.0,
    "neck": 0.8,
    "head": 0.6,
    "lShldr": 0.8,
    "rShldr": 0.8,
    "lHand": 1.2,  # 手部动作更重要
    "rHand": 1.2,
    "lFoot": 1.2,  # 脚部动作更重要
    "rFoot": 1.2,
}

<<<<<<< Updated upstream
=======
# ============================================================================
# 单树混合旋转索引配置
# ============================================================================

# 是否启用多树模式（False则使用传统单树模式）
# 即使包含旋转，我们也将其混合到同一棵树中
ENABLE_MULTI_TREE: bool = False

# 是否在单棵树中包含旋转增强的帧
INCLUDE_ROTATIONS: bool = True

# 旋转配置列表：定义每棵树的旋转参数
# 每个配置包含：axis（旋转轴：'x'/'y'/'z'）和 angle（角度：度）
ROTATION_CONFIGS: list[dict] = [
    {"axis": "z", "angle": 0},    # 原始坐标系
    {"axis": "z", "angle": 30},   # Z轴旋转30°
    {"axis": "z", "angle": 60},   # Z轴旋转60°
    {"axis": "y", "angle": 30},   # Y轴旋转30°
    {"axis": "x", "angle": 30},   # X轴旋转30°
]

# 候选集合并策略 (单树模式下不使用，但保留定义)
MERGE_STRATEGY: str = "vote"
MIN_VOTE_THRESHOLD: int = 2

# 多树训练时是否使用并行（单树模式下无效）
PARALLEL_TRAIN_TREES: bool = True

# 多树查询时是否使用并行（单树模式下无效）
PARALLEL_QUERY_TREES: bool = True
>>>>>>> Stashed changes
