"""
八叉树构建、训练与持久化相关函数（扁平化优化版）。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
import config
from data_structures import (
    BoundingBox,
    KeypointInput,
    MultiPointOctant,
    FrameMetadata,
    coerce_body_keypoints,
    compute_octant,
)
from flat_octree import FlatOctree
import pickle

# =============================================================================
# 临时构建节点 (仅用于构建过程)
# =============================================================================

def _get_active_joint_names(depth: int) -> Tuple[str, ...]:
    """根据深度获取当前活跃的关节对"""
    pair_idx = depth // config.PAIR_ITERATION_DEPTH
    if pair_idx >= len(config.JOINT_PAIRS):
        return config.JOINT_PAIRS[-1]
    return config.JOINT_PAIRS[pair_idx]

class _BuilderNode:
    """
    用于构建过程的临时节点类。
    仅在内存中存在，构建完成后转换为 FlatOctree 并销毁。
    """
    def __init__(self, depth: int, bboxes: Dict[str, BoundingBox], parent: Optional['_BuilderNode'] = None, path_code: str = ""):
        self.depth = depth
        self.parent = parent
        self.path_code = path_code # 空间路径编码：例如 "R-7-3-0"
        self.children: Dict[MultiPointOctant, _BuilderNode] = {}
        self.bboxes = bboxes
        self.frame_ids: List[str] = []

    def get_or_create_child(self, octants: MultiPointOctant) -> '_BuilderNode':
        if octants in self.children:
            return self.children[octants]
        
        # 获取当前层级活跃的关节
        active_names = _get_active_joint_names(self.depth)
        
        # 创建子节点包围盒
        # 默认继承父节点的所有包围盒
        child_bboxes = self.bboxes.copy()
        
        # 仅细分活跃关节的包围盒
        for idx, name in enumerate(active_names):
            bbox = self.bboxes[name]
            child_bboxes[name] = bbox.subdivide(octants[idx])
            
        # 生成子节点的路径编码
        octant_str = "".join(str(o) for o in octants)
        child_path = f"{self.path_code}-{octant_str}" if self.path_code else octant_str
            
        child = _BuilderNode(depth=self.depth + 1, bboxes=child_bboxes, parent=self, path_code=child_path)
        self.children[octants] = child
        return child

    def add_frame(self, frame_id: str):
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)

# =============================================================================
# 构建函数
# =============================================================================

def _create_root_bboxes() -> dict[str, BoundingBox]:
    """创建根节点的包围盒"""
    return {
        name: BoundingBox.from_tuple(bounds)
        for name, bounds in config.ROOT_BOUNDING_BOXES.items()
        if name in config.OCTREE_KEYPOINT_NAMES
    }

def create_root_node() -> _BuilderNode:
    """构建根节点"""
    return _BuilderNode(depth=0, bboxes=_create_root_bboxes(), parent=None, path_code="R") # R 代表 Root

def insert_frame(root: _BuilderNode, keypoints: KeypointInput, frame_id: str) -> None:
    """
    将单个帧插入八叉树。
    """
    body = coerce_body_keypoints(keypoints)
    keypoint_map = body.as_dict()

    current = root
    current.add_frame(frame_id)
    
    for _ in range(config.MAX_DEPTH):
        # 获取当前活跃关节
        active_names = _get_active_joint_names(current.depth)
        
        # 只为活跃关节计算octant
        octants = tuple(
            compute_octant(keypoint_map[name], current.bboxes[name])
            for name in active_names
        )
        current = current.get_or_create_child(octants)
        current.add_frame(frame_id)

# =============================================================================
# 扁平化转换与IO
# =============================================================================

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
    
    # 计算最大键宽度（最大关节对大小）
    max_key_width = max(len(pair) for pair in config.JOINT_PAIRS)
    
    # 2. 初始化数组
    flat = FlatOctree()
    flat.num_nodes = num_nodes
    flat.keypoint_names = kp_names
    
    flat.node_depth = np.zeros(num_nodes, dtype=np.int8)
    flat.node_parent_idx = np.full(num_nodes, -1, dtype=np.int32)
    flat.node_path_codes = np.zeros(num_nodes, dtype='U64') # 假设最大深度不会导致路径过长
    flat.bboxes = np.zeros((num_nodes, num_kps, 6), dtype=np.float32)
    
    # 子节点 CSR 数组
    total_children = sum(len(n.children) for n in nodes)
    flat.children_start = np.zeros(num_nodes + 1, dtype=np.int32)
    flat.children_keys = np.zeros((total_children, max_key_width), dtype=np.uint8)
    flat.children_indices = np.zeros(total_children, dtype=np.int32)
    
    # 帧ID 数组
    total_frames = sum(len(n.frame_ids) for n in nodes)
    flat.frame_ids_start = np.zeros(num_nodes + 1, dtype=np.int32)
    flat.frame_ids_data = np.empty(total_frames, dtype='U32')
    
    # 3. 填充数据
    child_ptr = 0
    frame_ptr = 0
    
    for i, node in enumerate(nodes):
        # 节点信息
        flat.node_depth[i] = node.depth
        if node.parent is not None:
            flat.node_parent_idx[i] = node_to_idx[node.parent]
            
        flat.node_path_codes[i] = node.path_code
            
        # 包围盒
        for k, name in enumerate(kp_names):
            bbox = node.bboxes[name]
            flat.bboxes[i, k, 0:3] = bbox.min_point
            flat.bboxes[i, k, 3:6] = bbox.max_point
            
        # 子节点
        flat.children_start[i] = child_ptr
        sorted_keys = sorted(node.children.keys())
        for key in sorted_keys:
            child_node = node.children[key]
            
            # 如果需要，填充键
            key_arr = np.zeros(max_key_width, dtype=np.uint8)
            key_arr[:len(key)] = key
            
            flat.children_keys[child_ptr] = key_arr
            flat.children_indices[child_ptr] = node_to_idx[child_node]
            child_ptr += 1
            
        # 帧ID
        flat.frame_ids_start[i] = frame_ptr
        for fid in node.frame_ids:
            flat.frame_ids_data[frame_ptr] = fid
            frame_ptr += 1
            
    flat.children_start[num_nodes] = child_ptr
    flat.frame_ids_start[num_nodes] = frame_ptr
    
    return flat

def save_tree(root: _BuilderNode, path: str, show_progress: bool = True) -> None:
    """
    保存树。
    会自动将构建节点转换为扁平化格式并保存为 .npz。
    """
    if show_progress:
        print("正在转换为扁平化结构...")
    flat = build_flat_tree(root)
    
    if show_progress:
        print(f"正在保存到 {path} ...")
    flat.save(path)

def load_tree(path: str, show_progress: bool = True) -> FlatOctree:
    """
    加载树。
    返回 FlatOctree 对象。
    """
    if show_progress:
        print(f"正在加载模型 {path} ...")
    return FlatOctree.load(path)

def save_metadata(metadata_list: List[FrameMetadata], path: str) -> None:
    """保存帧元数据列表到pkl文件"""
    with open(path, "wb") as file:
        pickle.dump(metadata_list, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_metadata(path: str) -> List[FrameMetadata]:
    """从pkl文件加载帧元数据列表"""
    with open(path, "rb") as file:
        return pickle.load(file)
