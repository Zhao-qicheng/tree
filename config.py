"""
系统级配置项。

该模块集中定义人体动作八叉树系统所需的常量、关键点列表、空间参数以及推理回退策略。
支持 Human3.6M 17关节格式（用于 FS-Jump3D 数据集）。
"""

from __future__ import annotations

from typing import Dict, Tuple

# ============================================================================
# 数据源配置
# ============================================================================

# 数据源类型："bvh" 或 "npy"
DATA_SOURCE_TYPE: str = "npy"

# FS-Jump3D 数据目录（NPY 格式）
FS_JUMP3D_DATA_DIR: str = "/home/u2023312269/数据集/FS-Jump3D-main/data/npy"

# 标准骨骼比例（基于数据集平均值，以 Hip->Neck 总长度为 100.0 时的比例）
# 格式: "parentIdx_childIdx": 比例值
STANDARD_BONE_RATIOS: Dict[str, float] = {
    '0_7': 0.4380, '7_8': 0.3804, '8_9': 0.1816, '9_10': 0.2238,
    '8_11': 0.2817, '11_12': 0.4463, '12_13': 0.4279,
    '8_14': 0.2866, '14_15': 0.4498, '15_16': 0.4589,
    '0_4': 0.1634, '4_5': 0.7955, '5_6': 0.7563,
    '0_1': 0.1634, '1_2': 0.8356, '2_3': 0.7395,
}

# 归一化参考基准长度（将 Hip->Neck 总长度设定为此值）
NORMALIZE_REFERENCE_LENGTH: float = 100.0

# ============================================================================
# Human3.6M 17关节配置
# ============================================================================

# 关键点名称，使用 Human3.6M 的 17 个关节
KEYPOINT_NAMES = (
    "hip",        # 0 - 髋部（原点）
    "rHip",       # 1 - 右髋
    "rKnee",      # 2 - 右膝
    "rAnkle",     # 3 - 右踝
    "lHip",       # 4 - 左髋
    "lKnee",      # 5 - 左膝
    "lAnkle",     # 6 - 左踝
    "spine",      # 7 - 脊柱
    "chest",      # 8 - 胸部
    "neck",       # 9 - 颈部
    "head",       # 10 - 头部
    "lShoulder",  # 11 - 左肩
    "lElbow",     # 12 - 左肘
    "lWrist",     # 13 - 左腕
    "rShoulder",  # 14 - 右肩
    "rElbow",     # 15 - 右肘
    "rWrist",     # 16 - 右腕
)

# 用于八叉树迭代的关键点（排除原点 hip）
OCTREE_KEYPOINT_NAMES = (
    "rHip",
    "rKnee",
    "rAnkle",
    "lHip",
    "lKnee",
    "lAnkle",
    "spine",
    "chest",
    "neck",
    "head",
    "lShoulder",
    "lElbow",
    "lWrist",
    "rShoulder",
    "rElbow",
    "rWrist",
)

# 关节对配置（16个关节分成8组，每组2个）
# 将关键点按对分组，每组在八叉树的一定深度范围内使用
JOINT_PAIRS = (
    ("chest", "spine"),       # 躯干核心
    ("neck", "head"),         # 头部
    ("lShoulder", "lElbow"),  # 左臂上部
    ("lWrist", "rShoulder"),  # 左手腕+右肩
    ("rElbow", "rWrist"),     # 右臂
    ("lHip", "lKnee"),        # 左腿上部
    ("lAnkle", "rHip"),       # 左踝+右髋
    ("rKnee", "rAnkle"),      # 右腿下部
)

# 每组关节对的迭代深度 n
# 前 n 层使用第一组，n+1 到 2n 层使用第二组，以此类推
PAIR_ITERATION_DEPTH: int = 2

# 八叉树相关配置
# 动态计算最大深度
MAX_DEPTH: int = len(JOINT_PAIRS) * PAIR_ITERATION_DEPTH

# 根节点包围盒尺寸参数
# 重定向归一化后，脊柱总长为100.0，整个人体动作范围约在 ±150 左右。
ROOT_BBOX_SIZE: float = 200.0  # 边长
ROOT_BBOX_HALF_SIZE: float = ROOT_BBOX_SIZE / 2.0

# 每个关键点在根节点的初始包围盒范围（min, max）
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
ACTION_LABELS: Dict[str, str] = {}

# JSON 持久化设置
JSON_FLOAT_PRECISION: int = 6
JSON_INDENT: int = 2

# 查询配置
TOP_K: int = 5  # 返回最相似的K个帧
EXACT_MATCH_EPSILON: float = 1  # 精确匹配的距离阈值
MIN_CANDIDATES: int = 1000  # 八叉树查询时的最小候选帧数量
BEAM_WIDTH: int = 1000  # 多分枝向下搜索时保留的候选节点数量
MAX_BACKTRACK_DEPTH: int = 11  # 候选不足时允许回溯的最大层级

# 相似度计算权重（Human3.6M 17关节）
# 提高手、脚、膝盖等关键末梢的权重，以捕捉动作细节
JOINT_WEIGHTS: Dict[str, float] = {
    "hip": 1.0,
    "rHip": 1.0,
    "rKnee": 1.0,
    "rAnkle": 1.0,
    "lHip": 1.0,
    "lKnee": 1.0,
    "lAnkle": 1.0,
    "spine": 1.0,
    "chest": 1.0,
    "neck": 1.0,
    "head": 1.0,
    "lShoulder": 1.0,
    "lElbow": 1.0,
    "lWrist": 1.0,
    "rShoulder": 1.0,
    "rElbow": 1.0,
    "rWrist": 1.0,
}

# ============================================================================
# 多树旋转索引配置
# ============================================================================

# 是否启用多树模式
ENABLE_MULTI_TREE: bool = False

# 旋转配置列表
ROTATION_CONFIGS: list[dict] = [
    {"axis": "z", "angle": 0},    # 原始坐标系
]

# 候选集合并策略
MERGE_STRATEGY: str = "vote"

# 投票法的最小投票数阈值
MIN_VOTE_THRESHOLD: int = 2

# 多树训练/查询并行
PARALLEL_TRAIN_TREES: bool = True
PARALLEL_QUERY_TREES: bool = True
