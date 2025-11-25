"""
查询脚本：加载已训练的八叉树模型并执行动作预测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_loader import load_keypoints_from_bvh
from inference import predict_action
from octree_builder import load_tree
import config

MODEL_PATH = Path("tree.json")  # 模型文件使用JSON格式
METADATA_PATH = Path("train_samples.json")
DEFAULT_BVH_FILE = Path("data/walk.bvh")
DEFAULT_QUERY_FRAME = 5


def print_main_joint_positions(keypoints: dict[str, np.ndarray]) -> None:
    """
    打印主要关节位置信息。
    
    参数:
        keypoints: 关键点字典，键为关节名称，值为3D坐标numpy数组
    """
    print("\n关键点位置（相对于Hips，单位：原始单位）:")
    print("-" * 80)
    
    # 按照配置的顺序打印关键点
    for joint_name in config.KEYPOINT_NAMES:
        if joint_name in keypoints:
            pos = keypoints[joint_name]
            # 格式化坐标显示
            print(f"  {joint_name:12s}: X={pos[0]:8.3f}, Y={pos[1]:8.3f}, Z={pos[2]:8.3f}")
        else:
            print(f"  {joint_name:12s}: <缺失>")
    
    print("-" * 80)


def print_combination_indices(combination_indices: list[str], label: str = "层") -> None:
    """
    打印组合索引列表，格式化显示每层的索引。
    
    参数:
        combination_indices: 组合索引字符串列表（如 ["227304", "123456", ...]）
        label: 标签前缀（如 "查询路径层" 或 "训练样本层"）
    """
    if not combination_indices:
        print(f"  {label}: <无索引>")
        return
    
    for idx, combo_idx in enumerate(combination_indices, start=1):
        if combo_idx:
            # 格式化显示：每2个字符一组，更容易阅读
            formatted = " ".join([combo_idx[i:i+2] for i in range(0, len(combo_idx), 2)])
            print(f"  {label} {idx:2d}: {combo_idx} ({formatted})")
        else:
            print(f"  {label} {idx:2d}: <空>")


def compute_weighted_distance(
    query_indices: list[str], 
    sample_indices: list[str]
) -> float:
    """
    计算两个组合索引列表之间的加权距离。
    
    距离计算规则：
    1. 对于每一层，如果索引相同则距离为0，否则根据差异计算距离
    2. 越深的层权重越大（距离根节点越远，权重越大）
    3. 如果列表长度不同，较短的列表在后续层视为最大距离
    
    参数:
        query_indices: 查询路径的组合索引列表
        sample_indices: 训练样本路径的组合索引列表
    
    返回:
        加权距离值（浮点数），值越小表示越相似
    """
    if not query_indices or not sample_indices:
        # 如果任一列表为空，返回最大距离
        return float('inf')
    
    total_distance = 0.0
    max_length = max(len(query_indices), len(sample_indices))
    
    # 对每一层计算距离
    for depth in range(max_length):
        # 权重：深度越深，权重越大（depth从0开始，所以+1）
        weight = (depth + 1) ** 2  # 使用平方权重，使得深层差异影响更大
        
        if depth >= len(query_indices) or depth >= len(sample_indices):
            # 如果某一层缺失，视为最大差异
            # 组合索引是6位数字（0-7），最大差异是每位数都不同，即6位*7=42
            layer_distance = 6 * 7  # 最大可能的层间距离
        else:
            query_idx = query_indices[depth]
            sample_idx = sample_indices[depth]
            
            if query_idx == sample_idx:
                layer_distance = 0.0
            else:
                # 计算两个索引字符串的差异
                # 每个索引是6位数字，每一位的取值范围是0-7
                layer_distance = 0.0
                max_len = max(len(query_idx), len(sample_idx))
                
                for pos in range(max_len):
                    q_char = int(query_idx[pos]) if pos < len(query_idx) else 0
                    s_char = int(sample_idx[pos]) if pos < len(sample_idx) else 0
                    # 计算每一位的差异
                    diff = abs(q_char - s_char)
                    layer_distance += diff
        
        total_distance += weight * layer_distance
    
    return total_distance


def _load_training_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        return []
    with open(metadata_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        return []
    return samples


def query(
    *,
    query_frame: int = DEFAULT_QUERY_FRAME,
    bvh_file: str | Path = DEFAULT_BVH_FILE,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> None:
    """对指定帧执行查询。"""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    bvh_file = Path(bvh_file)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先运行 train.py")

    root = load_tree(str(model_path))
    query_keypoints = load_keypoints_from_bvh(query_frame, str(bvh_file))

    print("=" * 80)
    print("开始执行查询...")
    print("=" * 80)
    print_main_joint_positions(query_keypoints)

    result = predict_action(root, query_keypoints)
    predicted_label = result.label
    print(f"\n预测结果: {predicted_label}")

    query_combination_indices = [
        entry.combination_index
        for entry in result.path[1:]  # 跳过根节点
        if entry.combination_index is not None
    ]

    print("\n查询路径组合索引:")
    print_combination_indices(query_combination_indices, "查询路径层")

    samples_meta = _load_training_metadata(metadata_path)
    if not samples_meta:
        print("\n未找到训练样本元数据，跳过候选匹配。")
        return

    print("\n候选匹配结果:")
    matched_samples: list[tuple[str, float]] = []
    for sample in samples_meta:
        combination_indices = sample.get("combination_indices", [])
        if not isinstance(combination_indices, list):
            continue
        sample_name = str(sample.get("sample_name", "unknown"))
        distance = compute_weighted_distance(
            query_combination_indices, list(map(str, combination_indices))
        )
        matched_samples.append((sample_name, distance))

    matched_samples.sort(key=lambda item: item[1])
    for sample_name, distance in matched_samples[:5]:
        print(f"候选 {sample_name} 距离 {distance:.6f}")
