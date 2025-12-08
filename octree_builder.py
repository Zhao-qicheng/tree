"""
八叉树构建、训练与持久化相关函数（帧检索系统版本）。
"""

from __future__ import annotations

import json
import pickle
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

<<<<<<< Updated upstream
=======
# =============================================================================
# 辅助函数
# =============================================================================

def get_active_pair_names(depth: int) -> Tuple[str, ...]:
    """根据深度获取当前活跃的关节对"""
    pair_idx = depth // config.PAIR_DEPTH
    if pair_idx >= len(config.JOINT_PAIRS):
        return ()
    return config.JOINT_PAIRS[pair_idx]

def encode_key(key: Tuple[int, ...]) -> int:
    """将 octant tuple 编码为单个字节"""
    if len(key) == 1:
        return key[0]
    elif len(key) == 2:
        return (key[0] << 3) | key[1]
    else:
        # Fallback or error, assume length 1 if empty or other
        return 0

# =============================================================================
# 临时构建节点 (仅用于构建过程)
# =============================================================================

class _BuilderNode:
    """
    用于构建过程的临时节点类。
    仅在内存中存在，构建完成后转换为 FlatOctree 并销毁。
    """
    def __init__(self, depth: int, bboxes: Dict[str, BoundingBox], parent: Optional['_BuilderNode'] = None):
        self.depth = depth
        self.parent = parent
        self.children: Dict[MultiPointOctant, _BuilderNode] = {}
        self.bboxes = bboxes
        self.frame_ids: List[str] = []

    def get_or_create_child(self, octants: MultiPointOctant) -> '_BuilderNode':
        if octants in self.children:
            return self.children[octants]
        
        # 获取当前活跃的关节名称
        active_names = get_active_pair_names(self.depth)
        
        # 创建子节点包围盒
        # 默认拷贝当前包围盒
        child_bboxes = self.bboxes.copy()
        
        # 仅细分活跃关节的包围盒
        for idx, name in enumerate(active_names):
            if idx < len(octants):
                bbox = self.bboxes[name]
                child_bboxes[name] = bbox.subdivide(octants[idx])
            
        child = _BuilderNode(depth=self.depth + 1, bboxes=child_bboxes, parent=self)
        self.children[octants] = child
        return child

    def add_frame(self, frame_id: str):
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)

# =============================================================================
# 构建函数
# =============================================================================
>>>>>>> Stashed changes

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
    
<<<<<<< Updated upstream
    combination_indices: list[str] = []

    for _ in range(config.MAX_DEPTH):
        # 只为用于八叉树的关键点计算octant（不包括hip原点）
        octants: MultiPointOctant = tuple(
=======
    for d in range(config.MAX_DEPTH):
        # 获取当前深度的活跃关节对
        active_names = get_active_pair_names(d)
        if not active_names:
            break
            
        # 计算活跃关节的 octant
        octants = tuple(
>>>>>>> Stashed changes
            compute_octant(keypoint_map[name], current.bboxes[name])
            for name in active_names
        )
<<<<<<< Updated upstream
        combination_index = compute_combination_index(octants)
        combination_indices.append(combination_index)
=======
>>>>>>> Stashed changes
        
        current = current.get_or_create_child(octants)
        current.add_frame(frame_id)

    return current, combination_indices

<<<<<<< Updated upstream
=======
def build_flat_tree(root: _BuilderNode) -> FlatOctree:
    """
    将构建树转换为扁平化八叉树。
    """
    # 1. BFS 遍历收集所有节点并分配索引
    nodes: List[_BuilderNode] = []
    queue = [root]
    
    # 节点对象 -> 索引 映射
    node_to_idx = {}
    
    while queue:
        node = queue.pop(0)
        idx = len(nodes)
        node_to_idx[node] = idx
        nodes.append(node)
        
        # 按排序后的key添加子节点，保证确定性
        sorted_keys = sorted(node.children.keys())
        for key in sorted_keys:
            queue.append(node.children[key])
            
    num_nodes = len(nodes)
    kp_names = list(config.OCTREE_KEYPOINT_NAMES)
    num_kps = len(kp_names)
    
    # 2. 初始化数组
    flat = FlatOctree()
    flat.num_nodes = num_nodes
    flat.keypoint_names = kp_names
    
    flat.node_depth = np.zeros(num_nodes, dtype=np.int8)
    flat.node_parent_idx = np.full(num_nodes, -1, dtype=np.int32)
    flat.bboxes = np.zeros((num_nodes, num_kps, 6), dtype=np.float32)
    
    # Children CSR arrays
    total_children = sum(len(n.children) for n in nodes)
    flat.children_start = np.zeros(num_nodes + 1, dtype=np.int32)
    # 修改为 1D 数组
    flat.children_keys = np.zeros(total_children, dtype=np.uint8)
    flat.children_indices = np.zeros(total_children, dtype=np.int32)
    
    # Frame IDs arrays
    total_frames = sum(len(n.frame_ids) for n in nodes)
    flat.frame_ids_start = np.zeros(num_nodes + 1, dtype=np.int32)
    flat.frame_ids_data = np.empty(total_frames, dtype='U32')
    
    # 3. 填充数据
    child_ptr = 0
    frame_ptr = 0
    
    for i, node in enumerate(nodes):
        # Node Info
        flat.node_depth[i] = node.depth
        if node.parent is not None:
            flat.node_parent_idx[i] = node_to_idx[node.parent]
            
        # Bboxes
        for k, name in enumerate(kp_names):
            bbox = node.bboxes[name]
            flat.bboxes[i, k, 0:3] = bbox.min_point
            flat.bboxes[i, k, 3:6] = bbox.max_point
            
        # Children
        flat.children_start[i] = child_ptr
        sorted_keys = sorted(node.children.keys())
        for key in sorted_keys:
            child_node = node.children[key]
            # Encode key
            flat.children_keys[child_ptr] = encode_key(key)
            flat.children_indices[child_ptr] = node_to_idx[child_node]
            child_ptr += 1
            
        # Frame IDs
        flat.frame_ids_start[i] = frame_ptr
        for fid in node.frame_ids:
            flat.frame_ids_data[frame_ptr] = fid
            frame_ptr += 1
            
    flat.children_start[num_nodes] = child_ptr
    flat.frame_ids_start[num_nodes] = frame_ptr
    
    return flat
>>>>>>> Stashed changes

def save_tree(root: ActionTreeNode, path: str) -> None:
    """
    使用pickle保存八叉树到二进制文件。
    
    参数:
        root: 八叉树根节点
        path: 保存路径（.tree文件）
    """
    with open(path, "wb") as file:
        pickle.dump(root, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_tree(path: str) -> ActionTreeNode:
    """
    从pickle文件加载八叉树。
    
    参数:
        path: 树文件路径（.tree文件）
    
    返回:
        八叉树根节点
    """
    with open(path, "rb") as file:
        return pickle.load(file)


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

