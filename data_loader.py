"""
从BVH文件加载数据并转换为八叉树系统所需的格式。
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import glob

import numpy as np

import sys

# 添加data目录到路径
sys.path.insert(0, str(Path(__file__).parent / "data"))
from data_frame import get_joint_position, get_frame_count

import config


def load_keypoints_from_bvh(frame_index: int, bvh_file: str = 'data/walk.bvh') -> Dict[str, np.ndarray]:
    """
    从BVH文件加载指定帧的6个关键点位置。
    
    参数:
        frame_index: 帧索引（从0开始）
        bvh_file: BVH文件路径
    
    返回:
        包含6个关键点位置的字典，键为BVH关节名称，值为numpy数组（单位：厘米）
    """
    keypoints = {}
    
    # 直接使用BVH文件中的关节名称
    for joint_name in config.KEYPOINT_NAMES:
        try:
            # 获取关节位置（相对于Hips），传递bvh_file参数
            pos = get_joint_position(
                frame_index, 
                joint_name, 
                bvh_file=bvh_file,
                relative_to_hips=True, 
                unit='original'
            )
            keypoints[joint_name] = pos
        except ValueError as e:
            raise ValueError(f"无法加载关节 {joint_name} 从文件 {bvh_file}: {e}")
    
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


def load_all_bvh_files(data_dir: str = "data/") -> List[str]:
    """
    扫描目录下所有.bvh文件。
    
    参数:
        data_dir: 数据目录路径
    
    返回:
        BVH文件路径列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")
    
    bvh_files = sorted(glob.glob(str(data_path / "*.bvh")))
    if not bvh_files:
        raise FileNotFoundError(f"在 {data_dir} 目录下没有找到BVH文件")
    
    return bvh_files


def get_bvh_frame_count(bvh_file: str) -> int:
    """
    获取BVH文件的总帧数。
    
    参数:
        bvh_file: BVH文件路径
    
    返回:
        总帧数
    """
    try:
        return get_frame_count(bvh_file)
    except Exception as e:
        raise ValueError(f"无法获取文件 {bvh_file} 的帧数: {e}")

