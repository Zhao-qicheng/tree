"""
八叉树构建、训练与持久化相关函数（帧检索系统版本）。
"""

from __future__ import annotations

import json
import pickle
from typing import Mapping, List
from pathlib import Path
import sys

from tqdm import tqdm

import config
from data_structures import (
    BoundingBox,
    KeypointInput,
    MultiPointOctant,
    FrameMetadata,
    coerce_body_keypoints,
    compute_octant,
    compute_combination_index,
)
from octree_node import ActionTreeNode


def _create_root_bboxes() -> dict[str, BoundingBox]:
    """创建根节点的包围盒，只包含用于八叉树迭代的关键点（不包括Hips）。"""
    return {
        name: BoundingBox.from_tuple(bounds)
        for name, bounds in config.ROOT_BOUNDING_BOXES.items()
        if name in config.OCTREE_KEYPOINT_NAMES
    }


def create_root_node() -> ActionTreeNode:
    """构建根节点，初始化所有关键点包围盒。"""
    return ActionTreeNode(depth=0, bboxes=_create_root_bboxes(), parent=None)


def insert_frame(root: ActionTreeNode, keypoints: KeypointInput, frame_id: str) -> tuple[ActionTreeNode, list[str]]:
    """
    将单个帧插入八叉树，返回最终叶节点和每层的组合索引路径。
    
    参数:
        root: 八叉树根节点
        keypoints: 关键点坐标
        frame_id: 帧的唯一标识
    
    返回:
        (叶节点, 组合索引列表): 每层的组合索引值
    """
    body = coerce_body_keypoints(keypoints)
    keypoint_map = body.as_dict()

    current = root
    current.add_frame(frame_id)
    
    combination_indices: list[str] = []

    for _ in range(config.MAX_DEPTH):
        # 只为用于八叉树的关键点计算octant（不包括hip原点）
        octants: MultiPointOctant = tuple(
            compute_octant(keypoint_map[name], current.bboxes[name])
            for name in config.OCTREE_KEYPOINT_NAMES
        )
        combination_index = compute_combination_index(octants)
        combination_indices.append(combination_index)
        
        current = current.get_or_create_child(octants)
        current.add_frame(frame_id)

    return current, combination_indices


def _detach_parent_links(root: ActionTreeNode,
                         show_progress: bool = False) -> list[tuple[ActionTreeNode, ActionTreeNode]]:
    """移除所有parent引用，并返回用于恢复的列表。"""
    stack = [root]
    detached: list[tuple[ActionTreeNode, ActionTreeNode]] = []

    # 使用进度条显示
    pbar = tqdm(desc="清除parent引用", unit="节点", disable=not show_progress)
    
    while stack:
        node = stack.pop()
        pbar.update(1)
        for _, child in node.iter_children():
            stack.append(child)
            if child.parent is not None:
                detached.append((child, child.parent))
                child.parent = None
    
    pbar.close()
    return detached


def _restore_parent_links(detached: list[tuple[ActionTreeNode, ActionTreeNode]],
                          show_progress: bool = False) -> None:
    """根据记录恢复parent引用。"""
    # 使用进度条显示
    pbar = tqdm(detached, desc="恢复parent引用", unit="引用", disable=not show_progress)
    
    for child, parent in pbar:
        child.parent = parent
    
    pbar.close()


def save_tree(root: ActionTreeNode, path: str, show_progress: bool = True) -> None:
    """
    使用pickle保存八叉树到二进制文件。

    保存前会临时清除所有parent引用以减少序列化体积和耗时，
    保存完成后再恢复引用。
    """
    sys.setrecursionlimit(20000)
    
    # 步骤1: 清除parent引用
    detached = _detach_parent_links(root, show_progress=show_progress)
    
    try:
        # 步骤2: 写入文件
        if show_progress:
            pbar = tqdm(total=1, desc="写入模型文件", unit="文件")
        with open(path, "wb") as file:
            pickle.dump(root, file, protocol=pickle.HIGHEST_PROTOCOL)
        if show_progress:
            pbar.update(1)
            pbar.close()
    finally:
        # 步骤3: 恢复parent引用
        _restore_parent_links(detached, show_progress=show_progress)


def _rebuild_parent_links(root: ActionTreeNode, show_progress: bool = False) -> None:
    """重新建立parent引用，支持查询阶段的回溯逻辑。"""
    stack = [root]
    
    # 使用进度条显示
    pbar = tqdm(desc="重建parent引用", unit="节点", disable=not show_progress)
    
    while stack:
        node = stack.pop()
        pbar.update(1)
        for _, child in node.iter_children():
            child.parent = node
            stack.append(child)
    
    pbar.close()


def load_tree(path: str, show_progress: bool = True) -> ActionTreeNode:
    """
    从pickle文件加载八叉树。
    
    参数:
        path: 树文件路径（.tree文件）
    
    返回:
        八叉树根节点
    """
    sys.setrecursionlimit(20000)
    
    if show_progress:
        pbar = tqdm(total=1, desc="读取模型文件", unit="文件")
    with open(path, "rb") as file:
        root: ActionTreeNode = pickle.load(file)
    if show_progress:
        pbar.update(1)
        pbar.close()

    _rebuild_parent_links(root, show_progress=show_progress)
    
    return root


def save_metadata(metadata_list: List[FrameMetadata], path: str) -> None:
    """
    保存帧元数据列表到pkl文件。
    
    参数:
        metadata_list: 帧元数据列表
        path: 保存路径（.pkl文件）
    """
    with open(path, "wb") as file:
        pickle.dump(metadata_list, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_metadata(path: str) -> List[FrameMetadata]:
    """
    从pkl文件加载帧元数据列表。
    
    参数:
        path: pkl文件路径
    
    返回:
        帧元数据列表
    """
    with open(path, "rb") as file:
        return pickle.load(file)

