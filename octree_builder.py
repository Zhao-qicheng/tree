"""
八叉树构建、训练与持久化相关函数（帧检索系统版本）。
"""

from __future__ import annotations

import json
import pickle
from collections import deque
from typing import Iterable, List, Mapping, Optional, Sequence
from pathlib import Path
import sys

import numpy as np

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
from flat_octree import FlatOctree


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


def save_tree(
    root: ActionTreeNode,
    path: str,
    show_progress: bool = True,
    *,
    use_flat: Optional[bool] = None,
) -> None:
    """
    保存八叉树到文件。

    默认根据路径后缀选择格式：.npz -> 扁平化格式，其余 -> 旧pickle格式。
    """
    if use_flat is None:
        use_flat = str(path).lower().endswith(".npz")

    if use_flat:
        if show_progress:
            print(f"[保存模型] 扁平化转换并写入 {path} ...")
        flat = convert_tree_to_flat(root, getattr(root, "keypoint_names", config.OCTREE_KEYPOINT_NAMES))
        flat.save(path)
        if show_progress:
            print(f"[保存模型] 完成，输出文件: {path}")
        return

    _save_tree_pickle(root, path, show_progress=show_progress)


def _save_tree_pickle(root: ActionTreeNode, path: str, show_progress: bool = True) -> None:
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


def load_tree(
    path: str,
    show_progress: bool = True,
    *,
    mmap_mode: Optional[str] = None,
) -> ActionTreeNode | FlatOctree:
    """
    加载八叉树文件。

    如果后缀为 .npz，则返回 FlatOctree；否则返回 ActionTreeNode。
    """
    use_flat = str(path).lower().endswith(".npz")
    if use_flat:
        if show_progress:
            print(f"[加载模型] 扁平化格式 -> {path}")
        flat = FlatOctree.load(path, mmap_mode=mmap_mode)
        if show_progress:
            print("[加载模型] 完成。")
        return flat

    return _load_tree_pickle(path, show_progress=show_progress)


def _load_tree_pickle(path: str, show_progress: bool = True) -> ActionTreeNode:
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


def convert_tree_to_flat(
    root: ActionTreeNode,
    keypoint_names: Optional[Sequence[str]] = None,
) -> FlatOctree:
    """
    将 ActionTreeNode 树转换为扁平化的 FlatOctree 结构。
    """
    if keypoint_names is None:
        keypoint_names = getattr(root, "keypoint_names", config.OCTREE_KEYPOINT_NAMES)
    keypoint_tuple = tuple(keypoint_names)
    num_keypoints = len(keypoint_tuple)

    queue: deque[ActionTreeNode] = deque([root])
    nodes: list[ActionTreeNode] = []
    node_index: dict[ActionTreeNode, int] = {}
    num_children_entries = 0
    num_frame_entries = 0
    max_frame_id_len = 1

    while queue:
        node = queue.popleft()
        idx = len(nodes)
        nodes.append(node)
        node_index[node] = idx

        child_items = list(node.iter_children())
        num_children_entries += len(child_items)
        for _, child in child_items:
            queue.append(child)

        num_frame_entries += len(node.frame_ids)
        for frame_id in node.frame_ids:
            if frame_id:
                max_frame_id_len = max(max_frame_id_len, len(frame_id))

    frame_id_dtype = f"U{max(8, max_frame_id_len)}"

    flat = FlatOctree.allocate(
        keypoint_names=keypoint_tuple,
        num_nodes=len(nodes),
        num_children_entries=num_children_entries,
        num_frame_entries=num_frame_entries,
        frame_id_dtype=frame_id_dtype,
    )

    child_offset = 0
    frame_offset = 0

    for idx, node in enumerate(nodes):
        parent_idx = -1 if node.parent is None else node_index[node.parent]
        flat.node_parent_idx[idx] = parent_idx
        flat.node_depth[idx] = node.depth

        for kp_idx, kp_name in enumerate(keypoint_tuple):
            bbox = node.bboxes.get(kp_name)
            if bbox is None:
                continue
            flat.bboxes[idx, kp_idx, :3] = bbox.min_point
            flat.bboxes[idx, kp_idx, 3:] = bbox.max_point

        flat.children_row_ptr[idx] = child_offset
        for octants, child in node.iter_children():
            flat.children_octants[child_offset, :] = np.asarray(octants, dtype=np.uint8)
            flat.children_indices[child_offset] = node_index[child]
            child_offset += 1

        flat.frame_ids_row_ptr[idx] = frame_offset
        for frame_id in node.frame_ids:
            flat.frame_ids_data[frame_offset] = frame_id
            frame_offset += 1

    flat.children_row_ptr[len(nodes)] = child_offset
    flat.frame_ids_row_ptr[len(nodes)] = frame_offset

    return flat


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
