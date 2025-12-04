"""
相似度计算与Top-K查询逻辑。
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

import config
from data_structures import FrameMetadata, KeypointInput, coerce_body_keypoints


# 预计算权重数组，用于快速计算
_OCTREE_WEIGHTS_ARRAY = np.zeros(len(config.KEYPOINT_NAMES), dtype=np.float32)
for i, name in enumerate(config.KEYPOINT_NAMES):
    # 只计算八叉树使用的关键点（不包括hip原点）
    if name in config.OCTREE_KEYPOINT_NAMES:
        _OCTREE_WEIGHTS_ARRAY[i] = config.JOINT_WEIGHTS.get(name, 1.0)


@dataclass
class SimilarityResult:
    """
    相似度查询结果。
    
    Attributes:
        frame_metadata: 帧元数据
        distance: 加权距离值（越小越相似）
        similarity_score: 相似度得分（0-1之间，1表示完全相同）
        is_exact_match: 是否为精确匹配
    """
    frame_metadata: FrameMetadata
    distance: float
    similarity_score: float
    is_exact_match: bool


def compute_weighted_distance(query_keypoints: Union[Dict[str, np.ndarray], np.ndarray], 
                              stored_keypoints: Union[Dict[str, np.ndarray], np.ndarray]) -> float:
    """
    计算两组关键点之间的加权欧氏距离。
    
    支持字典或numpy数组输入。如果输入为numpy数组，将使用向量化计算加速。
    """
    # 向量化路径
    if isinstance(query_keypoints, np.ndarray) and isinstance(stored_keypoints, np.ndarray):
        # 假设shape均为 (43, 3)
        diff = query_keypoints - stored_keypoints
        # axis=1 计算每个关节的欧氏距离 -> (43,)
        dists = np.linalg.norm(diff, axis=1)
        # 加权求和
        return float(np.dot(dists, _OCTREE_WEIGHTS_ARRAY))

    # 字典路径 (慢速回退)
    total_distance = 0.0
    
    # 只计算八叉树使用的关键点（不包括hip原点）
    for joint_name in config.OCTREE_KEYPOINT_NAMES:
        if joint_name not in query_keypoints or joint_name not in stored_keypoints:
            # 如果某个关节缺失，给予最大惩罚
            total_distance += 1000.0
            continue
        
        # 计算该关节的欧氏距离
        query_pos = query_keypoints[joint_name]
        stored_pos = stored_keypoints[joint_name]
        
        euclidean_dist = np.linalg.norm(query_pos - stored_pos)
        
        # 应用权重
        weight = config.JOINT_WEIGHTS.get(joint_name, 1.0)
        weighted_dist = euclidean_dist * weight
        
        total_distance += weighted_dist
    
    return total_distance


def compute_similarity_score(distance: float, max_distance: float = 1000.0) -> float:
    """
    将距离转换为相似度得分（0-1之间）。
    
    参数:
        distance: 加权距离值
        max_distance: 最大距离（用于归一化）
    
    返回:
        相似度得分，1表示完全相同，0表示完全不同
    """
    # 使用指数衰减函数
    # similarity = exp(-distance / scale)
    scale = max_distance / 10.0  # 调整衰减速度
    similarity = np.exp(-distance / scale)
    return float(similarity)


def is_exact_match(distance: float, epsilon: float = None) -> bool:
    """
    判断是否为精确匹配。
    
    参数:
        distance: 加权距离值
        epsilon: 精确匹配阈值（默认使用config.EXACT_MATCH_EPSILON）
    
    返回:
        如果距离小于epsilon，则认为是精确匹配
    """
    if epsilon is None:
        epsilon = config.EXACT_MATCH_EPSILON
    return distance < epsilon


def find_similar_frames(query_keypoints: KeypointInput,
                       metadata_list: List[FrameMetadata],
                       top_k: int = None,
                       enable_parallel: bool = True,
                       max_workers: Optional[int] = None) -> List[SimilarityResult]:
    """
    在元数据列表中查找与查询帧最相似的K个帧。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        metadata_list: 训练集的帧元数据列表
        top_k: 返回前K个最相似的帧（默认使用config.TOP_K）
    
    返回:
        按相似度排序的结果列表（距离从小到大）
    """
    if top_k is None:
        top_k = config.TOP_K
    
    # 预处理查询关键点为数组，加速计算
    body = coerce_body_keypoints(query_keypoints)
    query_array = np.zeros((len(config.KEYPOINT_NAMES), 3), dtype=np.float32)
    for i, name in enumerate(config.KEYPOINT_NAMES):
        query_array[i] = getattr(body, name)
    
    candidate_count = len(metadata_list)
    if candidate_count == 0:
        return []

    def compute_for_metadata(metadata: FrameMetadata) -> SimilarityResult:
        # 使用数组进行快速距离计算
        distance = compute_weighted_distance(query_array, metadata.get_keypoints_array())
        similarity_score = compute_similarity_score(distance)
        exact_match = is_exact_match(distance)
        return SimilarityResult(
            frame_metadata=metadata,
            distance=distance,
            similarity_score=similarity_score,
            is_exact_match=exact_match,
        )

    use_parallel = enable_parallel and candidate_count >= 32
    if use_parallel:
        workers = max_workers or (os.cpu_count() or 1)
        workers = min(workers, candidate_count)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(compute_for_metadata, metadata_list))
    else:
        results = [compute_for_metadata(metadata) for metadata in metadata_list]
    
    # 按距离排序（从小到大）
    results.sort(key=lambda x: x.distance)
    
    # 返回前K个
    return results[:top_k]


def find_similar_frames_in_candidates(query_keypoints: KeypointInput,
                                     metadata_list: List[FrameMetadata],
                                     candidate_frame_ids: List[str],
                                     top_k: int = None,
                                     enable_parallel: bool = True,
                                     max_workers: Optional[int] = None) -> List[SimilarityResult]:
    """
    在候选帧ID列表中查找与查询帧最相似的K个帧。
    
    这个函数用于在八叉树定位到的候选帧中进行精确匹配。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        metadata_list: 完整的帧元数据列表
        candidate_frame_ids: 候选帧ID列表（从八叉树中获取）
        top_k: 返回前K个最相似的帧
    
    返回:
        按相似度排序的结果列表
    """
    # 创建frame_id到metadata的映射
    metadata_dict = {m.frame_id: m for m in metadata_list}
    
    # 筛选出候选帧的元数据
    candidate_metadata = [
        metadata_dict[frame_id]
        for frame_id in candidate_frame_ids
        if frame_id in metadata_dict
    ]
    
    # 在候选帧中查找最相似的
    return find_similar_frames(
        query_keypoints,
        candidate_metadata,
        top_k,
        enable_parallel=enable_parallel,
        max_workers=max_workers,
    )


def compute_frame_distance_matrix(metadata_list: List[FrameMetadata]) -> np.ndarray:
    """
    计算所有帧之间的距离矩阵（用于分析和可视化）。
    
    参数:
        metadata_list: 帧元数据列表
    
    返回:
        距离矩阵，shape为(n_frames, n_frames)
    """
    n = len(metadata_list)
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = compute_weighted_distance(
                metadata_list[i].keypoints,
                metadata_list[j].keypoints
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # 对称矩阵
    
    return distance_matrix
