"""
NPY 数据加载模块：从 FS-Jump3D 数据集（Human3.6M 格式）加载 3D 姿态数据。

支持功能：
- 递归扫描 .npy 文件
- 朝向对齐（面朝 Y 轴正方向）
- 骨骼长度归一化
- 坐标中心化（相对于 Hip）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import numpy as np

import config

# Human3.6M 17关节索引
H36M_JOINT_NAMES = (
    "hip",        # 0 - 髋部（原点）
    "rHip",       # 1 - 右髋
    "rKnee",      # 2 - 右膝
    "rAnkle",     # 3 - 右踝
    "lHip",       # 4 - 左髋
    "lKnee",      # 5 - 左膝
    "lAnkle",     # 6 - 左踝
    "spine",      # 7 - 脊柱
    "chest",      # 8 - 胸部
    "neck",       # 9 - 颈部
    "head",       # 10 - 头部
    "lShoulder",  # 11 - 左肩
    "lElbow",     # 12 - 左肘
    "lWrist",     # 13 - 左腕
    "rShoulder",  # 14 - 右肩
    "rElbow",     # 15 - 右肘
    "rWrist",     # 16 - 右腕
)

# NPY 文件缓存
_npy_cache: Dict[str, np.ndarray] = {}


def load_npy_file(npy_file: str) -> np.ndarray:
    """
    加载 NPY 文件，支持缓存。
    
    参数:
        npy_file: NPY 文件路径
    
    返回:
        形状为 (帧数, 17, 3) 的数组
    """
    global _npy_cache
    
    npy_file = os.path.abspath(npy_file)
    
    if npy_file in _npy_cache:
        return _npy_cache[npy_file]
    
    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"NPY 文件不存在: {npy_file}")
    
    data = np.load(npy_file)
    _npy_cache[npy_file] = data
    return data


def clear_npy_cache():
    """清理 NPY 缓存"""
    global _npy_cache
    _npy_cache.clear()


def clear_specific_npy_file(npy_file: str):
    """清理特定文件的缓存"""
    global _npy_cache
    npy_file = os.path.abspath(npy_file)
    if npy_file in _npy_cache:
        del _npy_cache[npy_file]


# Human3.6M 骨骼拓扑结构（子节点 -> 父节点）
# 用于全量递归重定向
H36M_TOPOLOGY = {
    # 躯干
    7: 0, 8: 7, 9: 8, 10: 9,
    # 左腿
    4: 0, 5: 4, 6: 5,
    # 右腿
    1: 0, 2: 1, 3: 2,
    # 左臂
    11: 8, 12: 11, 13: 12,
    # 右臂
    14: 8, 15: 14, 16: 15
}


def align_orientation(frame: np.ndarray) -> np.ndarray:
    """
    朝向对齐：将骨盆向量旋转至 X 轴，使人体面朝 Y 轴正方向。
    
    步骤：
    1. 中心化：将 Hip (索引0) 移至原点
    2. 计算骨盆向量：从 LHip(4) 指向 RHip(1)
    3. 绕 Z 轴旋转使骨盆平行于 X 轴
    
    参数:
        frame: 形状为 (17, 3) 的单帧数据
    
    返回:
        对齐后的帧数据
    """
    # 1. 中心化
    hip = frame[0].copy()
    centered = frame - hip
    
    # 2. 计算骨盆向量：LHip(4) -> RHip(1)
    v_hip = centered[1] - centered[4]  # 右髋 - 左髋
    
    # 投影到 XY 平面
    v_hip_xy = np.array([v_hip[0], v_hip[1], 0])
    norm = np.linalg.norm(v_hip_xy)
    
    if norm < 1e-6:
        return centered  # 避免除零
    
    v_hip_norm = v_hip_xy / norm
    
    # 3. 计算与 X 轴的夹角 theta
    theta = np.arctan2(v_hip_norm[1], v_hip_norm[0])
    
    # 绕 Z 轴旋转 -theta 度
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # 应用旋转
    aligned = centered @ R.T
    return aligned


def normalize_skeleton(frame: np.ndarray) -> np.ndarray:
    """
    标准骨架重定向 (Standard Skeleton Retargeting)：
    
    使用向量法递归重构骨架。
    1. 根节点 (Hip, 0) 置于原点。
    2. 对于每个子节点，保持其相对于父节点的原始方向向量。
    3. 将方向向量乘以标准长度 (Standard Length)，得到新位置。
    
    公式：P_child_new = P_parent_new + Normalize(P_child_old - P_parent_old) * L_standard
    
    优点：
    - 完全保留动作的角度信息。
    - 强制所有骨骼长度符合标准比例。
    - 解决了不同身高/体型用户的动作在空间中不重合的问题。
    
    参数:
        frame: 形状为 (17, 3) 的原始帧数据
    
    返回:
        重定向后的帧数据 (17, 3)
    """
    base_len = getattr(config, 'NORMALIZE_REFERENCE_LENGTH', 100.0)
    ratios = getattr(config, 'STANDARD_BONE_RATIOS', {})
    
    new_frame = np.zeros_like(frame)
    # 1. 根节点归零
    # 注意：这里的 frame 假设已经是中心化甚至旋转对齐过的，但为了保险，
    # 我们再次强制将新骨架的 Hip 置于 0,0,0
    new_frame[0] = [0, 0, 0]
    
    # 2. 按照拓扑顺序计算 (确保父节点先被计算)
    # H36M 的索引顺序恰好大部分是父在子前，但为了严谨，我们显式从浅到深处理
    # 0 (Hip) 已处理
    # 第一层: 1, 4, 7
    # 第二层: 2, 5, 8
    # ...
    # 简单的做法是按索引排序处理，因为 H36M 索引大致符合层级，
    # 但严格来说应该用 BFS/DFS。考虑到 H36M 索引固定：
    # 0->1, 0->4, 0->7 是第一级。
    # 1->2, 4->5, 7->8 是第二级。
    # 8->9, 8->11, 8->14, 2->3, 5->6 是第三级。
    # ...
    # 为了简单且正确，我们定义一个处理顺序列表
    
    processing_order = [
         1, 4, 7,      # Level 1
         2, 5, 8,      # Level 2
         3, 6, 9, 11, 14, # Level 3
         10, 12, 15,   # Level 4
         13, 16        # Level 5
    ]
    
    for child_idx in processing_order:
        parent_idx = H36M_TOPOLOGY[child_idx]
        
        # 获取原始方向向量
        raw_vec = frame[child_idx] - frame[parent_idx]
        norm = np.linalg.norm(raw_vec)
        
        if norm < 1e-6:
            # 如果原始骨骼长度为0（重叠），则新骨骼也设为父节点位置
            new_frame[child_idx] = new_frame[parent_idx]
        else:
            # 归一化方向
            unit_vec = raw_vec / norm
            
            # 获取标准长度
            key = f"{parent_idx}_{child_idx}"
            ratio = ratios.get(key, 0.0) # 如果没找到比例，默认长度为0（还是有问题，应报错或给默认值）
            std_len = ratio * base_len
            
            # 计算新位置
            new_frame[child_idx] = new_frame[parent_idx] + unit_vec * std_len
            
    return new_frame


def load_keypoints_from_npy(frame_index: int, 
                            npy_file: str,
                            align: bool = True,
                            normalize: bool = False) -> Dict[str, np.ndarray]:
    """
    从 NPY 文件加载指定帧的关键点。
    
    参数:
        frame_index: 帧索引（从0开始）
        npy_file: NPY 文件路径
        align: 是否进行朝向对齐
        normalize: 是否进行骨骼归一化
    
    返回:
        包含17个关键点的字典，键为关节名称，值为 numpy 数组
    """
    data = load_npy_file(npy_file)
    
    if frame_index < 0 or frame_index >= data.shape[0]:
        raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {data.shape[0]-1}]")
    
    frame = data[frame_index].copy()  # (17, 3)
    
    # 1. 朝向对齐
    if align:
        frame = align_orientation(frame)
    else:
        # 仅中心化
        hip = frame[0].copy()
        frame = frame - hip
    
    # 2. 骨骼归一化
    if normalize:
        frame = normalize_skeleton(frame)
    
    # 3. 转换为字典
    keypoints = {}
    for i, name in enumerate(H36M_JOINT_NAMES):
        keypoints[name] = frame[i]
    
    return keypoints


def load_all_npy_files(data_dir: str) -> List[str]:
    """
    扫描目录下所有 .npy 文件，也支持扫描单个文件。
    
    参数:
        data_dir: 数据目录或单个文件路径
    
    返回:
        NPY 文件路径列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"路径 {data_dir} 不存在")
    
    # 如果是单个文件
    if data_path.is_file():
        if data_path.suffix.lower() == ".npy":
            return [str(data_path)]
        else:
            raise ValueError(f"文件 {data_dir} 不是 .npy 文件")
    
    # 递归搜索目录
    npy_files = sorted(glob.glob(str(data_path / "**/*.npy"), recursive=True))
    
    if not npy_files:
        raise FileNotFoundError(f"在 {data_dir} 目录下没有找到 NPY 文件")
    
    return npy_files


def get_npy_frame_count(npy_file: str) -> int:
    """
    获取 NPY 文件的总帧数。
    
    参数:
        npy_file: NPY 文件路径
    
    返回:
        总帧数
    """
    data = load_npy_file(npy_file)
    return data.shape[0]


def get_npy_metadata(npy_file: str) -> Dict[str, str]:
    """
    从文件路径解析元数据（运动员、跳跃类型）。
    
    假设路径格式：.../Skater_X/JumpType/filename.npy
    
    参数:
        npy_file: NPY 文件路径
    
    返回:
        包含 skater, jump_type, filename 的字典
    """
    path_parts = os.path.normpath(npy_file).split(os.sep)
    
    # 尝试解析
    try:
        # 找到 npy 目录的位置
        if 'npy' in path_parts:
            npy_idx = path_parts.index('npy')
            skater = path_parts[npy_idx + 1] if len(path_parts) > npy_idx + 1 else 'Unknown'
            jump_type = path_parts[npy_idx + 2] if len(path_parts) > npy_idx + 2 else 'Unknown'
        else:
            # 回退：使用倒数第三和倒数第二个目录
            skater = path_parts[-3] if len(path_parts) >= 3 else 'Unknown'
            jump_type = path_parts[-2] if len(path_parts) >= 2 else 'Unknown'
    except Exception:
        skater = 'Unknown'
        jump_type = 'Unknown'
    
    filename = Path(npy_file).stem
    
    return {
        'skater': skater,
        'jump_type': jump_type,
        'filename': filename
    }


def generate_frame_id_from_npy(npy_file: str, frame_index: int) -> str:
    """
    生成帧 ID（包含运动员、跳跃类型、文件名、帧号）。
    
    参数:
        npy_file: NPY 文件路径
        frame_index: 帧索引
    
    返回:
        帧 ID 字符串，格式：Skater_JumpType_Filename_frame_XXXX
    """
    meta = get_npy_metadata(npy_file)
    return f"{meta['skater']}_{meta['jump_type']}_{meta['filename']}_frame_{frame_index:04d}"


# 测试函数
if __name__ == "__main__":
    import sys
    
    test_dir = "c:/Users/86158/Desktop/数据集/FS-Jump3D-main/data/npy"
    
    print("=" * 60)
    print("NPY 加载器测试")
    print("=" * 60)
    
    # 测试文件扫描
    try:
        files = load_all_npy_files(test_dir)
        print(f"\n找到 {len(files)} 个 NPY 文件")
        print(f"前3个文件: {files[:3]}")
    except Exception as e:
        print(f"文件扫描失败: {e}")
        sys.exit(1)
    
    # 测试加载帧
    if files:
        test_file = files[0]
        print(f"\n测试文件: {test_file}")
        print(f"帧数: {get_npy_frame_count(test_file)}")
        
        # 加载第一帧
        kp = load_keypoints_from_npy(0, test_file)
        print(f"关节数: {len(kp)}")
        print(f"关节名称: {list(kp.keys())}")
        print(f"Hip 位置: {kp['hip']}")
        print(f"Chest 位置: {kp['chest']}")
        
        # 测试帧 ID 生成
        frame_id = generate_frame_id_from_npy(test_file, 0)
        print(f"帧 ID: {frame_id}")
    
    print("\n✅ 测试完成")
