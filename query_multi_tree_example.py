"""
多树查询脚本示例：演示如何使用多个旋转坐标系的八叉树进行查询。

这是一个示例脚本，展示如何扩展现有的query.py来支持多树查询。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

import numpy as np

from rotation_utils import create_default_rotation_configs, RotationConfig
from octree_builder import load_tree, load_metadata
from similarity import find_similar_frames_in_candidates, SimilarityResult
from data_structures import FrameMetadata
from octree_node import ActionTreeNode
from query import find_candidate_frames_from_tree
import config


def query_multi_tree(
    query_keypoints: dict[str, np.ndarray],
    model_dir: str = "models_multi/",
    rotation_configs: List[RotationConfig] = None,
    top_k: int = None,
    merge_strategy: str = "vote",
    min_vote_threshold: int = 2,
    verbose: bool = True
) -> List[SimilarityResult]:
    """
    使用多树进行查询。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        model_dir: 模型目录
        rotation_configs: 旋转配置列表
        top_k: 返回前K个最相似的帧
        merge_strategy: 合并策略 ("vote", "union", "weighted")
        min_vote_threshold: 最小投票数阈值
        verbose: 是否打印详细信息
    
    返回:
        相似度结果列表
    """
    if top_k is None:
        top_k = config.TOP_K
    
    if rotation_configs is None:
        rotation_configs = create_default_rotation_configs()
    
    num_trees = len(rotation_configs)
    
    if verbose:
        print("=" * 80)
        print(f"多树查询模式 - {num_trees}棵树")
        print("=" * 80)
    
    # 1. 加载所有树的模型（只加载一次元数据）
    trees: Dict[int, ActionTreeNode] = {}
    metadata_list: List[FrameMetadata] = None
    
    for cfg in rotation_configs:
        tree_path = str(Path(model_dir) / cfg.get_model_filename())
        metadata_path = str(Path(model_dir) / cfg.get_metadata_filename())
        
        if verbose:
            print(f"加载树 {cfg.tree_id}: {cfg}")
        
        trees[cfg.tree_id] = load_tree(tree_path, show_progress=False)
        
        # 只加载一次元数据（所有树的元数据frame_id相同）
        if metadata_list is None:
            metadata_list = load_metadata(metadata_path)
    
    if verbose:
        print(f"\n已加载 {num_trees} 棵树")
        print(f"训练集帧数: {len(metadata_list)}\n")
    
    # 2. 对每棵树进行查询，收集候选帧
    all_candidates: Dict[int, List[str]] = {}
    
    if verbose:
        print("步骤1: 从每棵树查找候选帧...")
    
    query_start = time.time()
    
    for cfg in rotation_configs:
        # 将查询关键点旋转到对应的坐标系
        rotated_query = cfg.rotate(query_keypoints)
        
        # 在该树中查找候选帧
        candidates = find_candidate_frames_from_tree(
            trees[cfg.tree_id],
            rotated_query,
            min_candidates=config.MIN_CANDIDATES
        )
        
        all_candidates[cfg.tree_id] = candidates
        
        if verbose:
            print(f"  树{cfg.tree_id}: {len(candidates)} 个候选帧")
    
    # 3. 合并候选帧
    if verbose:
        print(f"\n步骤2: 合并候选帧 (策略: {merge_strategy})...")
    
    merged_candidates = merge_candidates(
        all_candidates,
        strategy=merge_strategy,
        min_vote_threshold=min_vote_threshold,
        verbose=verbose
    )
    
    query_elapsed = time.time() - query_start
    
    if verbose:
        print(f"  合并后候选帧数: {len(merged_candidates)}")
        print(f"  八叉树查询用时: {query_elapsed:.4f} 秒\n")
    
    # 4. 在合并后的候选集中计算精确距离
    if verbose:
        print(f"步骤3: 计算精确距离并查找Top-{top_k}...")
    
    similarity_start = time.time()
    
    # 使用原始（未旋转）的查询关键点进行精确计算
    # 注意：metadata中存储的是旋转后的坐标，需要获取原始坐标
    # 为简化，这里假设使用树0（原始坐标系）的元数据
    results = find_similar_frames_in_candidates(
        query_keypoints,
        metadata_list,  # 使用原始坐标系的元数据
        merged_candidates,
        top_k
    )
    
    similarity_elapsed = time.time() - similarity_start
    
    if verbose:
        print(f"  精确计算用时: {similarity_elapsed:.4f} 秒")
        print(f"  总查询用时: {query_elapsed + similarity_elapsed:.4f} 秒\n")
    
    # 5. 打印结果
    if verbose:
        print("=" * 80)
        print(f"Top-{top_k} 查询结果:")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            metadata = result.frame_metadata
            filename = Path(metadata.bvh_file).name
            print(f"\n排名 {i}:")
            print(f"  文件: {filename}")
            print(f"  帧索引: {metadata.frame_index}")
            print(f"  距离: {result.distance:.6f}")
            print(f"  相似度: {result.similarity_score:.4f}")
        print("=" * 80)
    
    return results


def merge_candidates(
    all_candidates: Dict[int, List[str]],
    strategy: str = "vote",
    min_vote_threshold: int = 2,
    verbose: bool = False
) -> List[str]:
    """
    合并多棵树的候选帧。
    
    参数:
        all_candidates: {tree_id: [frame_ids]}
        strategy: 合并策略
        min_vote_threshold: 最小投票数
        verbose: 是否打印详细信息
    
    返回:
        合并后的候选帧ID列表
    """
    if strategy == "union":
        # 策略1: 并集（所有树的候选帧合并）
        merged_set: Set[str] = set()
        for candidates in all_candidates.values():
            merged_set.update(candidates)
        merged = list(merged_set)
        
        if verbose:
            print(f"  使用并集策略，候选帧: {len(merged)} 个")
        
        return merged
    
    elif strategy == "vote":
        # 策略2: 投票（至少在N棵树中出现）
        vote_counter = Counter()
        for candidates in all_candidates.values():
            for frame_id in candidates:
                vote_counter[frame_id] += 1
        
        # 过滤：只保留投票数 >= 阈值的帧
        merged = [
            frame_id for frame_id, count in vote_counter.items()
            if count >= min_vote_threshold
        ]
        
        if verbose:
            print(f"  使用投票策略（阈值≥{min_vote_threshold}）")
            print(f"  投票统计:")
            vote_dist = Counter(vote_counter.values())
            for votes, count in sorted(vote_dist.items(), reverse=True):
                print(f"    {votes}票: {count} 个帧")
        
        return merged
    
    elif strategy == "intersection":
        # 策略3: 交集（在所有树中都出现）
        if not all_candidates:
            return []
        
        merged_set = set(list(all_candidates.values())[0])
        for candidates in list(all_candidates.values())[1:]:
            merged_set &= set(candidates)
        
        merged = list(merged_set)
        
        if verbose:
            print(f"  使用交集策略，候选帧: {len(merged)} 个")
        
        return merged
    
    else:
        raise ValueError(f"未知的合并策略: {strategy}")


def demo_multi_tree_query():
    """演示多树查询功能。"""
    print("\n" + "=" * 80)
    print("多树查询演示")
    print("=" * 80)
    
    # 检查模型是否存在
    model_dir = Path("models_multi")
    if not model_dir.exists():
        print(f"\n错误: 模型目录 {model_dir} 不存在")
        print("请先运行 train_multi_tree_example.py 训练模型")
        return
    
    # 加载一个测试查询帧
    from data_loader import load_keypoints_from_bvh
    
    try:
        # 假设使用data目录下的示例数据
        test_bvh = "data/01_01.bvh"
        test_frame = 10
        
        print(f"\n查询帧: {test_bvh}, 帧索引 {test_frame}")
        
        query_keypoints = load_keypoints_from_bvh(test_frame, test_bvh)
        
        # 使用3棵树进行查询（演示用）
        configs = create_default_rotation_configs()[:3]
        
        results = query_multi_tree(
            query_keypoints=query_keypoints,
            model_dir="models_multi/",
            rotation_configs=configs,
            top_k=5,
            merge_strategy="vote",
            min_vote_threshold=2,
            verbose=True
        )
        
        print("\n✅ 多树查询演示完成！")
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_multi_tree_query()
