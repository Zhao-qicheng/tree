"""
训练与查询脚本共享的辅助函数。
"""

from __future__ import annotations


def format_joint_name_for_display(joint_name: str) -> str:
    """将系统关节名称转换为显示名称。"""
    mapping = {
        "left_wrist": "WristLeft",
        "right_wrist": "WristRight",
        "neck": "Neck",
        "left_ankle": "AnkleLeft",
        "right_ankle": "AnkleRight",
    }
    return mapping.get(joint_name, joint_name)


def print_main_joint_positions(keypoints: dict, sample_name: str = "") -> None:
    """打印主关节位置（不包括hips）。"""
    if sample_name:
        print(f"样本 {sample_name} 主关节位置:")
    else:
        print("查询主关节位置:")

    display_order = [
        "left_wrist",
        "right_wrist",
        "neck",
        "left_ankle",
        "right_ankle",
    ]
    for joint_name in display_order:
        if joint_name in keypoints:
            pos = keypoints[joint_name]
            display_name = format_joint_name_for_display(joint_name)
            print(f"  {display_name}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")


def print_combination_indices(
    combination_indices: list[str], prefix: str = "层"
) -> None:
    """打印每层的组合索引。"""
    for i, idx in enumerate(combination_indices, start=1):
        print(f"{prefix}{i}组合索引 {idx}")


def _layer_difference(a: str | None, b: str | None) -> int:
    """计算单层两个组合索引之间的节点差值之和。"""
    max_diff_per_digit = 7
    if a is None and b is None:
        return 0
    if a is None:
        return max_diff_per_digit * len(b)  # type: ignore[arg-type]
    if b is None:
        return max_diff_per_digit * len(a)
    length = min(len(a), len(b))
    diff = sum(abs(int(a[i]) - int(b[i])) for i in range(length))
    if len(a) != len(b):
        diff += max_diff_per_digit * abs(len(a) - len(b))
    return diff


def compute_weighted_distance(
    query_indices: list[str], sample_indices: list[str]
) -> int:
    """
    根据层数权重计算查询路径与样本路径之间的距离。

    总层数为 n，顶层（第0层）权重为 8^(n-1)，逐层递减，底层权重为 8^0。
    """
    if not query_indices and not sample_indices:
        return 0

    total_layers = max(len(query_indices), len(sample_indices))
    distance = 0
    for depth in range(total_layers):
        weight = 2 ** (total_layers - depth - 1)
        layer_query = query_indices[depth] if depth < len(query_indices) else None
        layer_sample = sample_indices[depth] if depth < len(sample_indices) else None
        layer_diff = _layer_difference(layer_query, layer_sample)
        distance += layer_diff * weight
    return distance


