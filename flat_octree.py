"""
扁平化八叉树存储结构。

使用 Structure of Arrays (SoA) 布局，利用 Numpy 数组存储整棵树，
实现零拷贝加载和极快的序列化速度。
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Union

class FlatOctree:
    """
    扁平化八叉树。
    
    所有数据存储为 Numpy 数组。
    """
    def __init__(self):
        self.num_nodes: int = 0
        self.keypoint_names: List[str] = []
        
        # ================= Node Data (N,) =================
        # 节点深度 (int8)
        self.node_depth: Optional[np.ndarray] = None 
        # 父节点索引 (int32), 根节点为 -1
        self.node_parent_idx: Optional[np.ndarray] = None
        
        # ================= Bounding Boxes (N, 15, 6) =================
        # 存储每个节点的所有关键点包围盒
        # 维度1: 节点索引
        # 维度2: 关键点索引 (对应 self.keypoint_names)
        # 维度3: [min_x, min_y, min_z, max_x, max_y, max_z]
        self.bboxes: Optional[np.ndarray] = None
        
        # ================= Children (CSR Format) =================
        # CSR (Compressed Sparse Row) 格式存储子节点
        # children_start[i] 到 children_start[i+1] 是节点 i 的子节点数据范围
        self.children_start: Optional[np.ndarray] = None # (N+1,) int32
        
        # 子节点对应的 Octant 组合键
        # 形状: (TotalChildren, 15) uint8
        # 每一行是一个长度为 15 的 octant 序列
        self.children_keys: Optional[np.ndarray] = None 
        
        # 子节点在 node 数组中的索引
        # 形状: (TotalChildren,) int32
        self.children_indices: Optional[np.ndarray] = None
        
        # ================= Frame IDs (CSR-like) =================
        # 存储每个节点关联的帧ID
        self.frame_ids_start: Optional[np.ndarray] = None # (N+1,) int32
        self.frame_ids_data: Optional[np.ndarray] = None # (TotalFrames,) string/U32
        
    def save(self, path: str) -> None:
        """
        保存为 .npz 文件。
        """
        np.savez_compressed(
            path,
            num_nodes=self.num_nodes,
            keypoint_names=self.keypoint_names,
            node_depth=self.node_depth,
            node_parent_idx=self.node_parent_idx,
            bboxes=self.bboxes,
            children_start=self.children_start,
            children_keys=self.children_keys,
            children_indices=self.children_indices,
            frame_ids_start=self.frame_ids_start,
            frame_ids_data=self.frame_ids_data
        )
        
    @classmethod
    def load(cls, path: str, mmap_mode: str = 'r') -> 'FlatOctree':
        """
        从 .npz 文件加载。
        
        Args:
            path: 文件路径
            mmap_mode: 内存映射模式 ('r', 'r+', 'c', None)
        """
        instance = cls()
        # allow_pickle=True is needed for string arrays (keypoint_names, frame_ids_data)
        # But for mmap_mode to work effectively on numerical data, we load carefully.
        # np.load with mmap_mode returns a NpzFile where arrays are mmap-ed if possible.
        
        data = np.load(path, mmap_mode=mmap_mode, allow_pickle=True)
        
        instance.num_nodes = int(data['num_nodes'])
        instance.keypoint_names = list(data['keypoint_names'])
        
        instance.node_depth = data['node_depth']
        instance.node_parent_idx = data['node_parent_idx']
        instance.bboxes = data['bboxes']
        
        instance.children_start = data['children_start']
        instance.children_keys = data['children_keys']
        instance.children_indices = data['children_indices']
        
        instance.frame_ids_start = data['frame_ids_start']
        instance.frame_ids_data = data['frame_ids_data']
        
        return instance

    def get_children(self, node_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定节点的子节点信息。
        
        Returns:
            (keys, indices)
            keys: (K, 15) uint8 array, octant combinations
            indices: (K,) int32 array, child node indices
        """
        start = self.children_start[node_idx]
        end = self.children_start[node_idx + 1]
        return self.children_keys[start:end], self.children_indices[start:end]

    def get_frame_ids(self, node_idx: int) -> np.ndarray:
        """
        获取指定节点的帧ID列表。
        
        Returns:
            (F,) string array
        """
        start = self.frame_ids_start[node_idx]
        end = self.frame_ids_start[node_idx + 1]
        return self.frame_ids_data[start:end]
