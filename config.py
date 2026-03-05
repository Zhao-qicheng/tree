"""
系统级配置项。

该模块集中定义人体动作八叉树系统所需的常量、关键点列表、空间参数以及推理回退策略。
支持 Human3.6M 17关节格式（用于 FS-Jump3D 数据集）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

# ============================================================================
# 路径解析辅助：自动推导项目根目录，支持 .env 文件覆盖
# ============================================================================
# 项目根目录 = config.py 所在目录
_PROJECT_ROOT = Path(__file__).resolve().parent

def _load_dotenv(env_path: Path) -> None:
    """极简 .env 解析，无需安装 python-dotenv。
    只处理 KEY=VALUE 格式，忽略注释行和空行。"""
    if not env_path.exists():
        return
    with open(env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"\'')
            # 仅在未被系统环境变量覆盖时才设置
            if key and key not in os.environ:
                os.environ[key] = value

# 加载 .env 文件（项目根目录下，不提交到 git）
_load_dotenv(_PROJECT_ROOT / '.env')

def _path(env_key: str, default_relative: str) -> str:
    """从环境变量读取路径，若未设置则使用项目根目录的相对路径作为默认值。"""
    val = os.environ.get(env_key, '')
    if val:
        return val
    return str(_PROJECT_ROOT / default_relative)

# ============================================================================
# 数据源配置
# ============================================================================

# 数据源类型："bvh" 或 "npy"
DATA_SOURCE_TYPE: str = "npy"

# FS-Jump3D 数据目录（NPY 格式，原始完整数据）
# 优先读取环境变量 FS_JUMP3D_DATA_DIR，默认为项目内 data/npy
FS_JUMP3D_DATA_DIR: str = _path('FS_JUMP3D_DATA_DIR', 'data/npy')

# 划分后的训练集与测试集目录（由 split_dataset.py 生成）
TRAIN_DATA_DIR: str = _path('TRAIN_DATA_DIR', 'data/npy_train')
TEST_DATA_DIR: str  = _path('TEST_DATA_DIR',  'data/npy_test')

# 标准骨骼比例（基于数据集平均值，以 Hip->Neck 总长度为 100.0 时的比例）
# 格式: "parentIdx_childIdx": 比例值
STANDARD_BONE_RATIOS: Dict[str, float] = {
    '0_1': 0.185711, # hip -> rHip
    '1_2': 0.874799, # rHip -> rKnee
    '2_3': 0.795995, # rKnee -> rAnkle
    '0_4': 0.185711, # hip -> lHip
    '4_5': 0.874799, # lHip -> lKnee
    '5_6': 0.795995, # lKnee -> lAnkle
    '0_7': 0.433722, # hip -> spine
    '7_8': 0.357584, # spine -> chest
    '8_9': 0.208694, # chest -> neck
    '9_10': 0.244348, # neck -> head
    '8_11': 0.297835, # chest -> lShoulder
    '11_12': 0.490605, # lShoulder -> lElbow
    '12_13': 0.470700, # lElbow -> lWrist
    '8_14': 0.297835, # chest -> rShoulder
    '14_15': 0.490605, # rShoulder -> rElbow
    '15_16': 0.470700, # rElbow -> rWrist
}

# 归一化参考基准长度（将 Hip->Neck 总长度设定为此值）
NORMALIZE_REFERENCE_LENGTH: float = 510.6

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
    # ("chest", "spine"),       # 躯干核心
    # ("neck", "head"),         # 头部
    ("spine", "head"), # 躯干核心
    
    ("lWrist", "rWrist"),     # 双手腕
    ("lAnkle", "rAnkle"),     # 双足踝
)

# 每组关节对的迭代深度 n
# 前 n 层使用第一组，n+1 到 2n 层使用第二组，以此类推
PAIR_ITERATION_DEPTH: int = 1

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
