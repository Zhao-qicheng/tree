"""
Skeleton NPZ 数据加载模块。

data/skeleton 目录下的 .npz 文件包含键 `reconstruction`，
形状为 (帧数, 17, 3)，关节顺序与 Human3.6M 17 关节一致。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from npy_loader import (
    H36M_JOINT_NAMES,
    align_orientation,
    normalize_skeleton,
)

SKELETON_NPZ_KEY = "reconstruction"

_npz_cache: Dict[str, np.ndarray] = {}


def load_skeleton_npz_file(npz_file: str) -> np.ndarray:
    """
    加载 skeleton NPZ 文件，支持缓存。

    返回:
        形状为 (帧数, 17, 3) 的数组
    """
    global _npz_cache

    npz_file = os.path.abspath(npz_file)

    if npz_file in _npz_cache:
        return _npz_cache[npz_file]

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"NPZ 文件不存在: {npz_file}")

    data = np.load(npz_file, allow_pickle=True)
    if SKELETON_NPZ_KEY not in data:
        raise KeyError(
            f"NPZ 文件 {npz_file} 缺少键 '{SKELETON_NPZ_KEY}'，"
            f"实际键: {list(data.keys())}"
        )

    array = np.asarray(data[SKELETON_NPZ_KEY], dtype=np.float64)
    data.close()

    if array.ndim != 3 or array.shape[1] != 17 or array.shape[2] != 3:
        raise ValueError(
            f"NPZ 文件 {npz_file} 的 reconstruction 形状应为 (N, 17, 3)，"
            f"实际为 {array.shape}"
        )

    _npz_cache[npz_file] = array
    return array


def clear_skeleton_npz_cache() -> None:
    """清理 NPZ 缓存"""
    global _npz_cache
    _npz_cache.clear()


def clear_specific_skeleton_npz_file(npz_file: str) -> None:
    """清理特定文件的缓存"""
    global _npz_cache
    npz_file = os.path.abspath(npz_file)
    if npz_file in _npz_cache:
        del _npz_cache[npz_file]


def load_keypoints_from_skeleton_npz(
    frame_index: int,
    npz_file: str,
    align: bool = True,
    normalize: bool = False,
) -> Dict[str, np.ndarray]:
    """从 skeleton NPZ 文件加载指定帧的关键点。"""
    data = load_skeleton_npz_file(npz_file)

    if frame_index < 0 or frame_index >= data.shape[0]:
        raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {data.shape[0] - 1}]")

    frame = data[frame_index].copy()

    if align:
        frame = align_orientation(frame)
    else:
        hip = frame[0].copy()
        frame = frame - hip

    if normalize:
        frame = normalize_skeleton(frame)

    return {name: frame[i] for i, name in enumerate(H36M_JOINT_NAMES)}


def load_all_skeleton_npz_files(data_dir: str) -> List[str]:
    """扫描目录下所有 skeleton .npz 文件，也支持单个文件。"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"路径 {data_dir} 不存在")

    if data_path.is_file():
        if data_path.suffix.lower() == ".npz":
            return [str(data_path)]
        raise ValueError(f"文件 {data_dir} 不是 .npz 文件")

    npz_files = sorted(glob.glob(str(data_path / "**/*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"在 {data_dir} 目录下没有找到 NPZ 文件")

    return npz_files


def get_skeleton_npz_frame_count(npz_file: str) -> int:
    """获取 skeleton NPZ 文件的总帧数。"""
    return load_skeleton_npz_file(npz_file).shape[0]


def generate_frame_id_from_skeleton_npz(npz_file: str, frame_index: int) -> str:
    """生成帧 ID，格式：{文件名}_frame_{帧号}。"""
    filename = Path(npz_file).stem
    return f"{filename}_frame_{frame_index:04d}"


def is_skeleton_npz_file(path: str) -> bool:
    """判断路径是否为 skeleton 姿态 NPZ 文件。"""
    return Path(path).suffix.lower() == ".npz"


def load_pose_sequence(file_path: str) -> np.ndarray:
    """
    统一加载姿态序列，返回 (N, J, 3)。

    自动识别 skeleton NPZ（键 reconstruction）与普通 NPY。
    """
    if is_skeleton_npz_file(file_path):
        return load_skeleton_npz_file(file_path)
    return np.load(file_path)
