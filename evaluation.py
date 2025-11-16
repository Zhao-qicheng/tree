"""
评估工具。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from data_structures import KeypointInput
from inference import predict_action
from octree_node import ActionTreeNode


def evaluate_accuracy(
    root: ActionTreeNode,
    samples: Sequence[Tuple[KeypointInput, str]],
) -> Dict[str, float]:
    """
    计算总体准确率。
    """
    total = len(samples)
    correct = 0
    for keypoints, expected in samples:
        result = predict_action(root, keypoints)
        if result.label == expected:
            correct += 1
    accuracy = correct / total if total else 0.0
    return {
        "total": float(total),
        "correct": float(correct),
        "accuracy": accuracy,
    }


def confusion_matrix(
    root: ActionTreeNode,
    samples: Sequence[Tuple[KeypointInput, str]],
) -> Dict[str, Dict[str, int]]:
    """
    构建混淆矩阵：真实标签 -> 预测标签 -> 次数。
    """
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for keypoints, expected in samples:
        result = predict_action(root, keypoints)
        predicted = result.label or "__unknown__"
        matrix[expected][predicted] += 1
    # 转换为普通 dict 便于序列化
    return {label: dict(counter) for label, counter in matrix.items()}

