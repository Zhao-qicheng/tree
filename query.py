"""
查询脚本：加载已训练的八叉树模型并执行动作预测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_loader import load_keypoints_from_bvh
from inference import predict_action
from octree_builder import load_tree
from app_utils import (
    compute_weighted_distance,
    print_combination_indices,
    print_main_joint_positions,
)

MODEL_PATH = Path("tree.pkl")
METADATA_PATH = Path("train_samples.json")
DEFAULT_BVH_FILE = Path("data/walk.bvh")
DEFAULT_QUERY_FRAME = 5


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
    matched_samples: list[tuple[str, int]] = []
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


if __name__ == "__main__":
    query()


