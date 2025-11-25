"""
推理与路径追踪逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import config
from data_structures import (
    KeypointInput,
    MultiPointOctant,
    coerce_body_keypoints,
    compute_octant,
    compute_combination_index,
)
from octree_node import ActionTreeNode


@dataclass(frozen=True)
class PredictionPathEntry:
    depth: int
    octants: Optional[MultiPointOctant]
    combination_index: Optional[str]
    resolved_label: Optional[str]
    total_samples: int


@dataclass(frozen=True)
class PredictionResult:
    label: Optional[str]
    path: List[PredictionPathEntry]
    terminated_at_depth: int
    fallback_triggered: bool


def predict_action(root: ActionTreeNode, keypoints: KeypointInput) -> PredictionResult:
    """
    根据输入关键点预测动作标签。

    若路径中断，则按照配置回退到最近祖先节点。
    """
    body = coerce_body_keypoints(keypoints)
    keypoint_map = body.as_dict()

    current = root
    path: List[PredictionPathEntry] = [
        PredictionPathEntry(
            depth=current.depth,
            octants=None,
            combination_index=None,
            resolved_label=current.resolved_label,
            total_samples=current.total_samples,
        )
    ]

    fallback_triggered = False
    for depth in range(config.MAX_DEPTH):
        octants: MultiPointOctant = tuple(
            compute_octant(keypoint_map[name], current.bboxes[name])
            for name in config.KEYPOINT_NAMES
        )
        combination_index = compute_combination_index(octants)
        child = current.get_child(octants)
        if child is None:
            fallback_triggered = True
            break
        current = child
        path.append(
            PredictionPathEntry(
                depth=current.depth,
                octants=octants,
                combination_index=combination_index,
                resolved_label=current.resolved_label,
                total_samples=current.total_samples,
            )
        )

    label = current.resolved_label
    if label is None and config.FALLBACK_STRATEGY == "nearest_ancestor":
        # 向上寻找最近拥有标签的祖先。
        node = current
        while node and label is None:
            label = node.resolved_label
            node = node.parent  # type: ignore[attr-defined]

    return PredictionResult(
        label=label,
        path=path,
        terminated_at_depth=current.depth,
        fallback_triggered=fallback_triggered,
    )

