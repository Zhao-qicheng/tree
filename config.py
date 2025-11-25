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

# 八叉树相关配置
MAX_DEPTH: int = 12

# 根节点包围盒尺寸参数（单位：厘米）
ROOT_BBOX_SIZE: float = 200.0  # 边长（-100到100）
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

