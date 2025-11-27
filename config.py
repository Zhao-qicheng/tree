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

