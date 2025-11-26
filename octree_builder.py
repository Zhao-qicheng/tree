"""
八叉树构建、训练与持久化相关函数（帧检索系统版本）。
"""

from __future__ import annotations

import json
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle
    JOBLIB_AVAILABLE = False
from typing import Mapping, List
from pathlib import Path

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


def save_tree(root: ActionTreeNode, path: str) -> None:
    """
    保存八叉树到二进制文件（优先使用joblib，回退到pickle）。
    
    joblib对numpy数组和大型对象的序列化更快，并支持压缩。
    
    参数:
        root: 八叉树根节点
        path: 保存路径（.tree文件）
    """
    if JOBLIB_AVAILABLE:
        # 使用joblib保存，compress=3提供平衡的压缩比和速度
        joblib.dump(root, path, compress=0)
    else:
        # 回退到pickle
        with open(path, "wb") as file:
            pickle.dump(root, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_tree(path: str) -> ActionTreeNode:
    """
    从文件加载八叉树（自动检测joblib或pickle格式）。
    
    参数:
        path: 树文件路径（.tree文件）
    
    返回:
        八叉树根节点
    """
    if JOBLIB_AVAILABLE:
        try:
            # 尝试用joblib加载
            return joblib.load(path)
        except:
            # 如果失败（可能是旧的pickle格式），回退到pickle
            with open(path, "rb") as file:
                return pickle.load(file)
    else:
        # 如果没有joblib，使用pickle
        with open(path, "rb") as file:
            return pickle.load(file)


def save_metadata(metadata_list: List[FrameMetadata], path: str) -> None:
    """
    保存帧元数据列表到文件（优先使用joblib）。
    
    参数:
        metadata_list: 帧元数据列表
        path: 保存路径（.pkl文件）
    """
    if JOBLIB_AVAILABLE:
        # 使用joblib保存，compress=3提供平衡的压缩比和速度
        joblib.dump(metadata_list, path, compress=0)
    else:
        # 回退到pickle
        with open(path, "wb") as file:
            pickle.dump(metadata_list, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_metadata(path: str) -> List[FrameMetadata]:
    """
    从文件加载帧元数据列表（自动检测格式）。
    
    参数:
        path: 文件路径
    
    返回:
        帧元数据列表
    """
    if JOBLIB_AVAILABLE:
        try:
            # 尝试用joblib加载
            return joblib.load(path)
        except:
            # 如果失败，回退到pickle
            with open(path, "rb") as file:
                return pickle.load(file)
    else:
        # 如果没有joblib，使用pickle
        with open(path, "rb") as file:
            return pickle.load(file)

