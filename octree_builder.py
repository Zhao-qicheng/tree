"""
八叉树构建、训练与持久化相关函数（帧检索系统版本）。
"""

from __future__ import annotations

import json
import pickle
from typing import Mapping, List, Iterable
from pathlib import Path
import sys

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


def _create_root_bboxes(keypoint_names: Iterable[str]) -> dict[str, BoundingBox]:
    """创建根节点的包围盒，只包含用于八叉树迭代的关键点（不包括Hips）。"""
    return {
        name: BoundingBox.from_tuple(bounds)
        for name, bounds in config.ROOT_BOUNDING_BOXES.items()
        if name in keypoint_names
    }


def create_root_node(keypoint_names: Optional[Iterable[str]] = None) -> ActionTreeNode:
    """构建根节点，初始化所有关键点包围盒。"""
    if keypoint_names is None:
        keypoint_names = config.OCTREE_KEYPOINT_NAMES
    keypoint_tuple = tuple(keypoint_names)
    return ActionTreeNode(
        depth=0,
        bboxes=_create_root_bboxes(keypoint_tuple),
        parent=None,
        keypoint_names=keypoint_tuple,
    )


def insert_frame(root: ActionTreeNode,
                 keypoints: KeypointInput,
                 frame_id: str) -> tuple[ActionTreeNode, list[str]]:
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

    keypoint_names = root.keypoint_names

    for _ in range(config.MAX_DEPTH):
        # 只为用于八叉树的关键点计算octant（不包括hip原点）
        octants: MultiPointOctant = tuple(
            compute_octant(keypoint_map[name], current.bboxes[name])
            for name in keypoint_names
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
    processed = 0
    progress_step = 10000

    while stack:
        node = stack.pop()
        processed += 1
        if show_progress and processed % progress_step == 0:
            print(f"    [保存模型] 已遍历 {processed} 个节点...")
        for _, child in node.iter_children():
            stack.append(child)
            if child.parent is not None:
                detached.append((child, child.parent))
                child.parent = None
    if show_progress:
        print(f"    [保存模型] 节点遍历完成，共 {processed} 个节点。")
    return detached


def _restore_parent_links(detached: list[tuple[ActionTreeNode, ActionTreeNode]],
                          show_progress: bool = False) -> None:
    """根据记录恢复parent引用。"""
    total = len(detached)
    progress_step = 10000
    for idx, (child, parent) in enumerate(detached, start=1):
        child.parent = parent
        if show_progress and idx % progress_step == 0:
            print(f"    [保存模型] 已恢复 {idx}/{total} 条引用...")
    if show_progress and total:
        print(f"    [保存模型] 引用恢复完成，共 {total} 条。")


def save_tree(root: ActionTreeNode, path: str, show_progress: bool = True) -> None:
    """
    使用pickle保存八叉树到二进制文件。
    
    保存前会临时清除所有parent引用以减少序列化体积和耗时，
    保存完成后再恢复引用。
    """
    sys.setrecursionlimit(20000)
    if show_progress:
        print(f"[保存模型] 步骤1/3: 清除 parent 引用...")
    detached = _detach_parent_links(root, show_progress=show_progress)
    try:
        if show_progress:
            print(f"[保存模型] 步骤2/3: 写入 {path} ...")
        with open(path, "wb") as file:
            pickle.dump(root, file, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        if show_progress:
            print("[保存模型] 步骤3/3: 恢复 parent 引用...")
        _restore_parent_links(detached, show_progress=show_progress)
        if show_progress:
            print(f"[保存模型] 完成，输出文件: {path}")


def _rebuild_parent_links(root: ActionTreeNode, show_progress: bool = False) -> None:
    """重新建立parent引用，支持查询阶段的回溯逻辑。"""
    stack = [root]
    processed = 0
    progress_step = 10000
    while stack:
        node = stack.pop()
        processed += 1
        for _, child in node.iter_children():
            child.parent = node
            stack.append(child)
        if show_progress and processed % progress_step == 0:
            print(f"    [加载模型] 已恢复 {processed} 个节点的父引用...")
    if show_progress:
        print(f"    [加载模型] 父引用恢复完成，共 {processed} 个节点。")


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
        print(f"[加载模型] 步骤1/2: 正在读取 {path} ...")
    with open(path, "rb") as file:
        root: ActionTreeNode = pickle.load(file)

    if show_progress:
        print("[加载模型] 步骤2/2: 恢复 parent 引用...")
    _rebuild_parent_links(root, show_progress=show_progress)
    if show_progress:
        print("[加载模型] 完成。")
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

