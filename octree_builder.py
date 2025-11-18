"""
八叉树构建、训练与持久化相关函数。
"""

from __future__ import annotations

import json
from typing import Mapping

import config
from data_structures import (
    BoundingBox,
    KeypointInput,
    MultiPointOctant,
    coerce_body_keypoints,
    compute_octant,
    compute_combination_index,
)
from octree_node import ActionTreeNode


def _create_root_bboxes() -> dict[str, BoundingBox]:
    return {
        name: BoundingBox.from_tuple(bounds)
        for name, bounds in config.ROOT_BOUNDING_BOXES.items()
    }


def create_root_node() -> ActionTreeNode:
    """构建根节点，初始化所有关键点包围盒。"""
    return ActionTreeNode(depth=0, bboxes=_create_root_bboxes(), parent=None)


def insert_sample(root: ActionTreeNode, keypoints: KeypointInput, label: str) -> tuple[ActionTreeNode, list[str]]:
    """
    将单个样本插入八叉树，返回最终叶节点和每层的组合索引路径。
    
    返回:
        (叶节点, 组合索引列表): 每层的组合索引值
    """
    body = coerce_body_keypoints(keypoints)
    keypoint_map = body.as_dict()

    current = root
    current.record_sample(label)
    
    combination_indices: list[str] = []

    for _ in range(config.MAX_DEPTH):
        octants: MultiPointOctant = tuple(
            compute_octant(keypoint_map[name], current.bboxes[name])
            for name in config.KEYPOINT_NAMES
        )
        combination_index = compute_combination_index(octants)
        combination_indices.append(combination_index)
        
        current = current.get_or_create_child(octants)
        current.record_sample(label)

    return current, combination_indices


def finalize_tree(node: ActionTreeNode) -> None:
    """递归解析所有节点标签。"""
    for _, child in node.iter_children():
        finalize_tree(child)
    node.resolve_label()


def tree_to_dict(root: ActionTreeNode) -> dict:
    return root.to_serializable()


def tree_from_dict(data: dict) -> ActionTreeNode:
    return ActionTreeNode.from_serializable(data)


def save_tree(root: ActionTreeNode, path: str) -> None:
    data = tree_to_dict(root)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=config.JSON_INDENT,
            ensure_ascii=False,
        )


def load_tree(path: str) -> ActionTreeNode:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return tree_from_dict(data)

