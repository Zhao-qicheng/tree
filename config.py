"""
系统级配置项。

该模块集中定义人体动作八叉树系统所需的常量、关键点列表、空间参数以及推理回退策略。
"""

from __future__ import annotations

from typing import Dict, Tuple

# 关键点名称，直接使用BVH文件中的关节名称
KEYPOINT_NAMES = (
    "Hips",
    "LeftHand",
    "RightHand",
    "Neck",
    "LeftFoot",
    "RightFoot",
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

