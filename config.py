"""
系统级配置项。

该模块集中定义人体动作八叉树系统所需的常量、关键点列表、空间参数以及推理回退策略。
"""

from __future__ import annotations

from typing import Dict, Tuple

# 关键点名称，使用BVH文件中的16个关键关节
# 这些关节将用于帧检索系统
KEYPOINT_NAMES = (
    "hip",        # 原点，始终为 [0, 0, 0]
    "chest",      # 胸部
    "neck",       # 颈部
    "head",       # 头部
    "lShldr",     # 左肩
    "lForeArm",   # 左前臂
    "lHand",      # 左手
    "rShldr",     # 右肩
    "rForeArm",   # 右前臂
    "rHand",      # 右手
    "lThigh",     # 左大腿
    "lShin",      # 左小腿
    "lFoot",      # 左脚
    "rThigh",     # 右大腿
    "rShin",      # 右小腿
    "rFoot",      # 右脚
)

# 用于八叉树迭代的关键点（排除原点 hip）
OCTREE_KEYPOINT_NAMES = (
    "chest",
    "neck",
    "head",
    "lShldr",
    "lForeArm",
    "lHand",
    "rShldr",
    "rForeArm",
    "rHand",
    "lThigh",
    "lShin",
    "lFoot",
    "rThigh",
    "rShin",
    "rFoot",
)

# 关节对配置
# 将关键点按对分组，每组在八叉树的一定深度范围内使用
JOINT_PAIRS = (
    ("chest", "neck"),
    ("head", "lShldr"),
    ("lForeArm", "lHand"),
    ("rShldr", "rForeArm"),
    ("rHand", "lThigh"),
    ("lShin", "lFoot"),
    ("rThigh", "rShin"),
    ("rFoot",),  # 最后一个单独一组
)

# 每组关节对的迭代深度 n
# 前 n 层使用第一组，n+1 到 2n 层使用第二组，以此类推
PAIR_ITERATION_DEPTH: int = 2

# 八叉树相关配置
# 动态计算最大深度
MAX_DEPTH: int = len(JOINT_PAIRS) * PAIR_ITERATION_DEPTH

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
EXACT_MATCH_EPSILON: float = 1  # 精确匹配的距离阈值
MIN_CANDIDATES: int = 1000  # 八叉树查询时的最小候选帧数量（约为总数的1%）
BEAM_WIDTH: int = 100  # 多分枝向下搜索时保留的候选节点数量
MAX_BACKTRACK_DEPTH: int = 3  # 候选不足时允许回溯的最大层级

# 相似度计算权重（可根据关节重要性调整）
JOINT_WEIGHTS: Dict[str, float] = {
    "hip": 1.0,
    "chest": 1.0,
    "neck": 0.8,
    "head": 0.6,
    "lShldr": 0.8,
    "lForeArm": 1.0,  # 前臂动作对手势识别重要
    "lHand": 1.2,     # 手部动作更重要
    "rShldr": 0.8,
    "rForeArm": 1.0,  # 前臂动作对手势识别重要
    "rHand": 1.2,
    "lThigh": 1.1,    # 大腿对步态识别重要
    "lShin": 1.1,     # 小腿对步态识别重要
    "lFoot": 1.2,     # 脚部动作更重要
    "rThigh": 1.1,    # 大腿对步态识别重要
    "rShin": 1.1,     # 小腿对步态识别重要
    "rFoot": 1.2,
}

# ============================================================================
# 多树旋转索引配置
# ============================================================================

# 是否启用多树模式（False则使用传统单树模式，但会在单树中包含旋转后的数据）
ENABLE_MULTI_TREE: bool = False

# 旋转配置列表：定义每棵树的旋转参数
# 在单树模式下，这些旋转将作为数据增强应用到同一棵树中
ROTATION_CONFIGS: list[dict] = [
    {"axis": "z", "angle": 0},    # 树0: 原始坐标系
    {"axis": "z", "angle": 30},   # 树1: Z轴旋转30°
    {"axis": "z", "angle": 60},   # 树2: Z轴旋转60°
    {"axis": "y", "angle": 30},   # 树3: Y轴旋转30°（改变重力方向）
    {"axis": "x", "angle": 30},   # 树4: X轴旋转30°
]

# 候选集合并策略
# - "vote": 投票法（至少在N棵树中出现）- 推荐
# - "union": 并集（所有树的候选合并）
# - "intersection": 交集（在所有树中都出现）
MERGE_STRATEGY: str = "vote"

# 投票法的最小投票数阈值（仅当MERGE_STRATEGY="vote"时有效）
# 推荐值：2-3（在至少2-3棵树中出现的帧才被认为是候选）
MIN_VOTE_THRESHOLD: int = 2

# 多树训练时是否使用并行（加速训练）
PARALLEL_TRAIN_TREES: bool = True

# 多树查询时是否使用并行（加速查询）
PARALLEL_QUERY_TREES: bool = True

