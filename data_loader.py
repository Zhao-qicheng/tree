"""
从BVH文件加载数据并转换为八叉树系统所需的格式。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import sys
from pathlib import Path

# 添加data目录到路径
sys.path.insert(0, str(Path(__file__).parent / "data"))
from data_frame import get_joint_position

# BVH关节名称到系统关节名称的映射
BVH_JOINT_MAPPING: Dict[str, str] = {
    'Hips': 'hips',
    'LeftHand': 'left_wrist',
    'RightHand': 'right_wrist',
    'Neck': 'neck',
    'LeftFoot': 'left_ankle',
    'RightFoot': 'right_ankle',
}


def load_keypoints_from_bvh(frame_index: int, bvh_file: str = 'data/walk.bvh') -> Dict[str, np.ndarray]:
    """
    从BVH文件加载指定帧的6个关键点位置。
    
    参数:
        frame_index: 帧索引（从0开始）
        bvh_file: BVH文件路径
    
    返回:
        包含6个关键点位置的字典，键为系统关节名称，值为numpy数组（单位：厘米）
    """
    keypoints = {}
    
    for bvh_name, sys_name in BVH_JOINT_MAPPING.items():
        try:
            # 获取关节位置（相对于hips，单位：厘米/original）
            pos = get_joint_position(frame_index, bvh_name, relative_to_hips=True, unit='original')
            keypoints[sys_name] = pos
        except ValueError as e:
            raise ValueError(f"无法加载关节 {bvh_name} (系统名称: {sys_name}): {e}")
    
    return keypoints


def load_multiple_frames(frame_indices: List[int], bvh_file: str = 'data/walk.bvh') -> List[Dict[str, np.ndarray]]:
    """
    从BVH文件加载多个帧的关键点数据。
    
    参数:
        frame_indices: 帧索引列表
        bvh_file: BVH文件路径
    
    返回:
        关键点字典列表
    """
    return [load_keypoints_from_bvh(idx, bvh_file) for idx in frame_indices]

