"""
帧检索查询脚本：加载扁平化模型并执行相似帧查询。
支持 BVH 和 NPY（Human3.6M）格式。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set, Union
from collections import Counter

import numpy as np

from octree_builder import load_tree, load_metadata
from similarity import find_similar_frames_in_candidates, SimilarityResult
from data_structures import coerce_body_keypoints, compute_octant, FrameMetadata, BoundingBox
import config
import json
import os


_MODEL_CACHE: dict[Tuple[str, str], Tuple[FlatOctree, List[FrameMetadata]]] = {}


def _compute_node_distances_batch(
    flat: FlatOctree,
    node_indices: np.ndarray, # (K,) int32
    query_keypoints: dict[str, np.ndarray]
) -> np.ndarray: # (K,) float32
    """
    批量计算节点与查询关键点的距离（向量化实现）。
    """
    if len(node_indices) == 0:
        return np.array([], dtype=np.float32)

    # 1. 收集包围盒: (K, 15, 6)
    node_bboxes = flat.bboxes[node_indices]
    
    # 2. 计算中心点: (K, 15, 3)
    centers = (node_bboxes[..., :3] + node_bboxes[..., 3:]) * 0.5
    
    # 3. 准备查询向量: (15, 3)
    query_vec = np.zeros((len(flat.keypoint_names), 3), dtype=np.float32)
    valid_mask = np.zeros(len(flat.keypoint_names), dtype=bool)
    
    for i, name in enumerate(flat.keypoint_names):
        if name in query_keypoints:
            query_vec[i] = query_keypoints[name]
            valid_mask[i] = True
            
    # 4. 计算距离
    # 差值: (K, 15, 3)
    # query_vec 需要进行广播
    diff = centers - query_vec # (K, 15, 3)
    dist_sq = np.sum(diff**2, axis=2) # (K, 15)
    dist = np.sqrt(dist_sq)
    
    # 掩盖无效的关键点
    dist = dist * valid_mask # (K, 15)
    
    count = np.sum(valid_mask)
    if count == 0:
        return np.full(len(node_indices), float('inf'), dtype=np.float32)
        
    total_dist = np.sum(dist, axis=1) # (K,)
    return total_dist / count


def find_candidate_frames_from_tree(tree: FlatOctree,
                                    query_keypoints: dict[str, np.ndarray],
                                    min_candidates: int = None) -> list[str]:
    """
    使用扁平化八叉树查找候选帧。
    """
    if min_candidates is None:
        min_candidates = config.MIN_CANDIDATES
    
    # 标准化查询关键点
    body = coerce_body_keypoints(query_keypoints)
    keypoint_dict = body.as_dict()
    
    # 波束搜索 (Beam Search)
    beam_width = max(1, getattr(config, "BEAM_WIDTH", 4))
    
    # 从根节点开始 (索引 0)
    current_nodes = np.array([0], dtype=np.int32)
    
    # 记录遇到的叶子节点
    leaf_nodes = []
    
    for _ in range(config.MAX_DEPTH):
        next_nodes = []
        
        # 扩展所有当前节点
        # 优化：如果我们实现了批量 get_children，可以批量处理，
        # 但这里我们遍历波束节点（通常很小，如 100）
        
        all_children_indices = []
        
        for node_idx in current_nodes:
            keys, children_idx = tree.get_children(node_idx)
            
            if len(children_idx) == 0:
                leaf_nodes.append(node_idx)
                continue
            
            all_children_indices.extend(children_idx)
            
        if not all_children_indices:
            break
            
        all_children_indices = np.array(all_children_indices, dtype=np.int32)
        
        # 一次性计算所有子节点的得分
        scores = _compute_node_distances_batch(tree, all_children_indices, keypoint_dict)
        
        # 选择 top-k
        if len(all_children_indices) > beam_width:
            top_k_idx = np.argsort(scores)[:beam_width]
            current_nodes = all_children_indices[top_k_idx]
        else:
            current_nodes = all_children_indices
            
    # 候选节点是 current_nodes + leaf_nodes
    candidate_node_indices = list(current_nodes) + leaf_nodes
    
    # 收集帧
    candidate_frame_ids = []
    seen_frames = set()
    visited_nodes = set()
    
    def collect_from_node(idx):
        if idx in visited_nodes: return
        visited_nodes.add(idx)
        fids = tree.get_frame_ids(idx)
        for fid in fids:
            if fid not in seen_frames:
                candidate_frame_ids.append(fid)
                seen_frames.add(fid)

    for idx in candidate_node_indices:
        collect_from_node(idx)
        
    # 回溯 (Backtracking)
    backtrack_depth = 0
    max_backtrack_depth = getattr(config, "MAX_BACKTRACK_DEPTH", 2)
    
    current_layer = candidate_node_indices
    
    while (len(candidate_frame_ids) < min_candidates and
           current_layer and
           backtrack_depth < max_backtrack_depth):
        parents = []
        for idx in current_layer:
            parent_idx = tree.node_parent_idx[idx]
            if parent_idx != -1 and parent_idx not in visited_nodes:
                collect_from_node(parent_idx)
                parents.append(parent_idx)
        current_layer = parents
        backtrack_depth += 1
    
    return candidate_frame_ids


def _normalize_query_input(keypoints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    对查询输入进行标准化处理：
    1. 旋转对齐 (Align Orientation)
    2. 骨架重定向 (Skeleton Retargeting)
    """
    from npy_loader import align_orientation, normalize_skeleton
    
    # Dict -> Array
    names = config.KEYPOINT_NAMES
    try:
        frame_array = np.array([keypoints[name] for name in names]) # (17, 3)
    except KeyError as e:
        print(f"Warning: Query keypoints missing joint {e}, skipping normalization.")
        return keypoints

    # 1. Align
    if getattr(config, "ENABLE_ALIGN", True):
        aligned_array = align_orientation(frame_array)
    else:
        aligned_array = frame_array
    
    # 2. Normalize
    if getattr(config, "ENABLE_NORMALIZE", True):
        normalized_array = normalize_skeleton(aligned_array)
    else:
        normalized_array = aligned_array
    
    # Array -> Dict
    return {name: normalized_array[i] for i, name in enumerate(names)}


def merge_candidates(all_candidates: Dict[int, List[str]],
                    strategy: str = None,
                    min_vote_threshold: int = None,
                    verbose: bool = False) -> List[str]:
    """合并多棵树的候选帧"""
    if strategy is None:
        strategy = config.MERGE_STRATEGY
    if min_vote_threshold is None:
        min_vote_threshold = config.MIN_VOTE_THRESHOLD
    
    if strategy == "union":
        merged_set: Set[str] = set()
        for candidates in all_candidates.values():
            merged_set.update(candidates)
        merged = list(merged_set)
        if verbose:
            print(f"  使用并集策略，候选帧: {len(merged)} 个")
        return merged
    
    elif strategy == "vote":
        vote_counter = Counter()
        for candidates in all_candidates.values():
            for frame_id in candidates:
                vote_counter[frame_id] += 1
        
        merged = [
            frame_id for frame_id, count in vote_counter.items()
            if count >= min_vote_threshold
        ]
        if verbose:
            print(f"  使用投票策略（阈值≥{min_vote_threshold}）")
        return merged
    
    elif strategy == "intersection":
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


def _load_model_once(model_tree_path: str,
                     model_metadata_path: str,
                     use_cache: bool = True,
                     show_progress: bool = True) -> tuple[FlatOctree, List[FrameMetadata], bool]:
    """加载模型，支持缓存"""
    cache_key = (
        str(Path(model_tree_path).resolve()),
        str(Path(model_metadata_path).resolve()),
    )
    if use_cache and cache_key in _MODEL_CACHE:
        tree, metadata_list = _MODEL_CACHE[cache_key]
        return tree, metadata_list, True

    tree = load_tree(model_tree_path, show_progress=show_progress)
    metadata_list = load_metadata(model_metadata_path)

    if use_cache:
        _MODEL_CACHE[cache_key] = (tree, metadata_list)

    return tree, metadata_list, False


def query_single_tree(query_keypoints: dict[str, np.ndarray],
                     model_tree_path: str,
                     model_metadata_path: str,
                     rotation_config: Optional[RotationConfig] = None,
                     top_k: int = None,
                     verbose: bool = True,
                     tree_instance: Optional[FlatOctree] = None,
                     metadata_instance: Optional[List[FrameMetadata]] = None,
                     use_cache: bool = True,
                     enable_parallel: bool = True,
                     parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """在单树中查询相似帧"""
    if top_k is None:
        top_k = config.TOP_K
    
    # 1. 加载模型
    load_elapsed = 0.0
    if tree_instance is not None and metadata_instance is not None:
        tree = tree_instance
        metadata_list = metadata_instance
        from_cache = True
    else:
        load_start_time = time.time()
        tree, metadata_list, from_cache = _load_model_once(
            model_tree_path,
            model_metadata_path,
            use_cache=use_cache,
            show_progress=False,
        )
        load_elapsed = time.time() - load_start_time
    
    # 2. 应用旋转
    query_kp = query_keypoints
    if rotation_config:
        query_kp = rotation_config.rotate(query_keypoints)
    
    # 3. 使用八叉树查找候选帧
    tree_search_start_time = time.time()
    candidate_frame_ids = find_candidate_frames_from_tree(tree, query_kp)
    tree_search_elapsed = time.time() - tree_search_start_time
    
    # 4. 在候选帧中查找Top-K相似帧
    similarity_start_time = time.time()
    results = find_similar_frames_in_candidates(
        query_keypoints,
        metadata_list,
        candidate_frame_ids,
        top_k,
        enable_parallel=enable_parallel,
        max_workers=parallel_workers,
    )
    similarity_elapsed = time.time() - similarity_start_time
    
    return results, candidate_frame_ids, load_elapsed, tree_search_elapsed, similarity_elapsed


def _load_leaf_labels(label_path: str = "data/leaf_labels.json") -> dict:
    """加载叶节点动作标签"""
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_node_index_for_query(tree: FlatOctree, query_keypoints: dict[str, np.ndarray]) -> int:
    """
    找到查询帧落入的八叉树节点索引（尽可能深入）。
    """
    body = coerce_body_keypoints(query_keypoints)
    keypoint_map = body.as_dict()

    current_node = 0  # 从根节点开始
    for depth in range(config.MAX_DEPTH):
        from octree_builder import _get_active_joint_names
        active_names = _get_active_joint_names(depth)
        
        # 寻找匹配的子节点
        # 注意：这里需要计算 octants，但扁平化树没有直接存储 bbox 对象，
        # 我们需要根据树中的数据动态创建 BoundingBox
        
        octants = tuple(
            compute_octant(keypoint_map[name], BoundingBox.from_tuple(
                (tuple(tree.bboxes[current_node, tree.keypoint_names.index(name), 0:3]),
                 tuple(tree.bboxes[current_node, tree.keypoint_names.index(name), 3:6]))
            ))
            for name in active_names
        )
        
        keys, indices = tree.get_children(current_node)
        found_next = False
        for i, key in enumerate(keys):
            if tuple(key[:len(octants)]) == octants:
                current_node = indices[i]
                found_next = True
                break
        
        if not found_next:
            break
    
    return current_node


def query_frame(query_keypoints: dict[str, np.ndarray],
               model_tree_path: str = "models/model.npz",
               model_metadata_path: str = "models/model.pkl",
               top_k: int = None,
               verbose: bool = True,
               *,
               tree_instance: Optional[FlatOctree] = None,
               metadata_instance: Optional[List[FrameMetadata]] = None,
               use_cache: bool = True,
               enable_parallel: bool = True,
               parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """执行帧检索查询"""
    if top_k is None:
        top_k = config.TOP_K
        
    # [NEW] 对输入帧进行标准化 (重定向)
    query_keypoints_norm = _normalize_query_input(query_keypoints)
    
    # 0. 加载模型（如果没传实例）
    if tree_instance is None:
        tree, metadata_list, _ = _load_model_once(model_tree_path, model_metadata_path, use_cache=use_cache, show_progress=False)
    else:
        tree = tree_instance
        metadata_list = metadata_instance

    # 1. 查找查询帧所属节点并获取标签 (使用归一化后的数据)
    leaf_idx = get_node_index_for_query(tree, query_keypoints_norm)
    labels = _load_leaf_labels()
    action_name = labels.get(str(leaf_idx))

    # 2. 执行检索 (使用归一化后的数据)
    results, candidate_ids, load_elapsed, tree_elapsed, sim_elapsed = query_single_tree(
        query_keypoints=query_keypoints_norm,
        model_tree_path=model_tree_path,
        model_metadata_path=model_metadata_path,
        rotation_config=None,
        top_k=top_k,
        verbose=False,
        tree_instance=tree,
        metadata_instance=metadata_list,
        use_cache=use_cache,
        enable_parallel=enable_parallel,
        parallel_workers=parallel_workers,
    )
    
    if verbose:
        print("=" * 80)
        print("帧检索查询系统（关节对分组八叉树 + 旋转增强 + 骨架重定向）")
        print("=" * 80)
        
        # 输出动作分类信息
        print(f"\n[动作识别结果]")
        if action_name:
            print(f"  查询帧识别为: {action_name} (匹配叶节点 #{leaf_idx})")
        else:
            print(f"  查询帧匹配叶节点 #{leaf_idx} (暂无动作名称记录)")
            
        print(f"\n[检索性能统计]")
        print(f"  候选帧数量: {len(candidate_ids)}")
        print(f"  八叉树查询用时: {tree_elapsed:.4f} 秒")
        print(f"  精确计算用时: {sim_elapsed:.4f} 秒")
        print(f"  总查询用时: {tree_elapsed + sim_elapsed:.4f} 秒\n")
        print(f"Top-{top_k}最相似的帧:")
    
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print_result(result, i)
        print("-" * 80)
    
    return results


def find_frames_in_same_node(query_keypoints: dict[str, np.ndarray],
                             model_tree_path: str = "models/model.npz",
                             model_metadata_path: str = "models/model.pkl",
                             verbose: bool = True,
                             *,
                             tree_instance: Optional[FlatOctree] = None,
                             metadata_instance: Optional[List[FrameMetadata]] = None,
                             use_cache: bool = True) -> List[str]:
    """
    找到与查询帧处于同一个八叉树叶子节点的所有帧。
    """
    # [NEW] 对输入帧进行标准化
    query_keypoints_norm = _normalize_query_input(query_keypoints)
    
    # 1. 加载模型
    if tree_instance is not None:
        tree = tree_instance
    else:
        tree, _, _ = _load_model_once(model_tree_path, model_metadata_path, use_cache=use_cache, show_progress=False)
    
    # 2. 沿着树向下走到底 (使用归一化后的数据)
    current_node = get_node_index_for_query(tree, query_keypoints_norm)
            
    # 3. 获取该节点的所有帧
    frame_ids = list(tree.get_frame_ids(current_node))
    
    if verbose:
        print(f"\n同节点查询完成！")
        print(f"  叶子节点索引: {current_node}")
        
        # 尝试输出标签
        labels = _load_leaf_labels()
        name = labels.get(str(current_node))
        if name:
            print(f"  动作名称: {name}")
        
        print(f"  同节点帧数量: {len(frame_ids)}")
        print(f"  帧ID列表: {frame_ids[:10]}{'...' if len(frame_ids) > 10 else ''}")
        
    return frame_ids


def query_same_node_frames_from_npy(npy_file: str,
                                   frame_index: int,
                                   model_tree_path: str = "models/model.npz",
                                   model_metadata_path: str = "models/model.pkl",
                                   verbose: bool = True,
                                   *,
                                   tree_instance: Optional[FlatOctree] = None,
                                   use_cache: bool = True) -> List[str]:
    """从 NPY 文件加载查询帧并查找同节点帧"""
    from npy_loader import load_keypoints_from_npy
    keypoints = load_keypoints_from_npy(frame_index, npy_file)
    return find_frames_in_same_node(
        keypoints,
        model_tree_path,
        model_metadata_path,
        verbose,
        tree_instance=tree_instance,
        use_cache=use_cache
    )


def query_same_node_frames_from_skeleton_npz(npz_file: str,
                                             frame_index: int,
                                             model_tree_path: str = "models/model.npz",
                                             model_metadata_path: str = "models/model.pkl",
                                             verbose: bool = True,
                                             *,
                                             tree_instance: Optional[FlatOctree] = None,
                                             use_cache: bool = True) -> List[str]:
    """从 skeleton NPZ 文件加载查询帧并查找同节点帧"""
    from skeleton_npz_loader import load_keypoints_from_skeleton_npz
    keypoints = load_keypoints_from_skeleton_npz(frame_index, npz_file)
    return find_frames_in_same_node(
        keypoints,
        model_tree_path,
        model_metadata_path,
        verbose,
        tree_instance=tree_instance,
        use_cache=use_cache
    )


def query_same_node_frames_from_bvh(bvh_file: str,
                                   frame_index: int,
                                   model_tree_path: str = "models/model.npz",
                                   model_metadata_path: str = "models/model.pkl",
                                   verbose: bool = True,
                                   *,
                                   tree_instance: Optional[FlatOctree] = None,
                                   use_cache: bool = True) -> List[str]:
    """从 BVH 文件加载查询帧并查找同节点帧"""
    from data_loader import load_keypoints_from_bvh
    keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
    return find_frames_in_same_node(
        keypoints,
        model_tree_path,
        model_metadata_path,
        verbose,
        tree_instance=tree_instance,
        use_cache=use_cache
    )


def print_result(result: SimilarityResult, rank: int) -> None:
    metadata = result.frame_metadata
    filename = Path(metadata.source_file).name
    match_flag = "[精确匹配]" if result.is_exact_match else ""
    print(f"\n排名 {rank}: {match_flag}")
    print(f"  文件名: {filename}")
    print(f"  帧索引: {metadata.frame_index}")
    print(f"  帧ID: {metadata.frame_id}")
    print(f"  距离值: {result.distance:.6f}")
    print(f"  相似度: {result.similarity_score:.4f}")


def query_from_bvh(bvh_file: str,
                  frame_index: int,
                  model_tree_path: str = "models/model.npz",
                  model_metadata_path: str = "models/model.pkl",
                  top_k: int = None,
                  verbose: bool = True,
                  *,
                  tree_instance: Optional[FlatOctree] = None,
                  metadata_instance: Optional[List[FrameMetadata]] = None,
                  use_cache: bool = True,
                  enable_parallel: bool = True,
                  parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    from data_loader import load_keypoints_from_bvh
    
    if verbose:
        print(f"\n从BVH文件加载查询帧...")
        print(f"  文件: {bvh_file}")
        print(f"  帧索引: {frame_index}")
    
    keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
    return query_frame(
        keypoints,
        model_tree_path,
        model_metadata_path,
        top_k,
        verbose,
        tree_instance=tree_instance,
        metadata_instance=metadata_instance,
        use_cache=use_cache,
        enable_parallel=enable_parallel,
        parallel_workers=parallel_workers,
    )


def query_from_npy(npy_file: str,
                   frame_index: int,
                   model_tree_path: str = "models/model.npz",
                   model_metadata_path: str = "models/model.pkl",
                   top_k: int = None,
                   verbose: bool = True,
                   *,
                   tree_instance: Optional[FlatOctree] = None,
                   metadata_instance: Optional[List[FrameMetadata]] = None,
                   use_cache: bool = True,
                   enable_parallel: bool = True,
                   parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """
    从 NPY 文件加载查询帧并执行检索。
    
    参数:
        npy_file: NPY 文件路径
        frame_index: 帧索引
        model_tree_path: 模型树文件路径
        model_metadata_path: 元数据文件路径
        top_k: 返回前 K 个结果
        verbose: 是否打印详细信息
    
    返回:
        相似度结果列表
    """
    from npy_loader import load_keypoints_from_npy
    
    if verbose:
        print(f"\n从NPY文件加载查询帧...")
        print(f"  文件: {npy_file}")
        print(f"  帧索引: {frame_index}")
    
    keypoints = load_keypoints_from_npy(frame_index, npy_file)
    return query_frame(
        keypoints,
        model_tree_path,
        model_metadata_path,
        top_k,
        verbose,
        tree_instance=tree_instance,
        metadata_instance=metadata_instance,
        use_cache=use_cache,
        enable_parallel=enable_parallel,
        parallel_workers=parallel_workers,
    )


def query_from_skeleton_npz(npz_file: str,
                            frame_index: int,
                            model_tree_path: str = "models/model.npz",
                            model_metadata_path: str = "models/model.pkl",
                            top_k: int = None,
                            verbose: bool = True,
                            *,
                            tree_instance: Optional[FlatOctree] = None,
                            metadata_instance: Optional[List[FrameMetadata]] = None,
                            use_cache: bool = True,
                            enable_parallel: bool = True,
                            parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """从 skeleton NPZ 文件加载查询帧并执行检索。"""
    from skeleton_npz_loader import load_keypoints_from_skeleton_npz

    if verbose:
        print(f"\n从 skeleton NPZ 文件加载查询帧...")
        print(f"  文件: {npz_file}")
        print(f"  帧索引: {frame_index}")

    keypoints = load_keypoints_from_skeleton_npz(frame_index, npz_file)
    return query_frame(
        keypoints,
        model_tree_path,
        model_metadata_path,
        top_k,
        verbose,
        tree_instance=tree_instance,
        metadata_instance=metadata_instance,
        use_cache=use_cache,
        enable_parallel=enable_parallel,
        parallel_workers=parallel_workers,
    )


def interactive_mode(model_tree_path: str,
                     model_metadata_path: str,
                     top_k: Optional[int] = None,
                     verbose: bool = True) -> None:
    print("=" * 80)
    print("交互式查询模式")
    print("=" * 80)
    print("提示: 输入 \"<BVH路径> <帧索引>\", 或输入 q 退出。")

    load_start = time.time()
    tree, metadata_list, from_cache = _load_model_once(
        model_tree_path,
        model_metadata_path,
        use_cache=True,
        show_progress=True,
    )
    load_elapsed = 0.0 if from_cache else time.time() - load_start
    print(f"\n模型已加载，帧数: {len(metadata_list)}，耗时: {load_elapsed:.2f} 秒")

    while True:
        try:
            print("\n查询> ", end="", flush=True)
            user_input = sys.stdin.readline()
            if not user_input:
                break
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input: continue
        if user_input.lower() in {"q", "quit", "exit"}: break

        parts = user_input.split()
        if len(parts) < 2:
            print("输入格式错误，请使用: <BVH路径> <帧索引>")
            continue

        frame_index_str = parts[-1]
        bvh_file = " ".join(parts[:-1])

        try:
            frame_index = int(frame_index_str)
        except ValueError:
            print("帧索引必须为整数。")
            continue

        if not Path(bvh_file).exists():
            print(f"BVH文件不存在: {bvh_file}")
            continue

        try:
            from data_loader import load_keypoints_from_bvh
            keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
            query_frame(
                keypoints,
                model_tree_path=model_tree_path,
                model_metadata_path=model_metadata_path,
                top_k=top_k,
                verbose=verbose,
                tree_instance=tree,
                metadata_instance=metadata_list,
                use_cache=True,
            )
        except Exception as exc:
            print(f"查询失败: {exc}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="帧检索查询")
    parser.add_argument("--bvh-file", help="BVH文件路径")
    parser.add_argument("--npy-file", help="NPY文件路径（Human3.6M格式）")
    parser.add_argument("--skeleton-npz-file", help="Skeleton NPZ 文件路径（reconstruction 数组）")
    parser.add_argument("--frame-index", type=int, help="帧索引")
    parser.add_argument("--model-tree", default="models/model.npz", help="树模型文件路径")
    parser.add_argument("--model-metadata", default="models/model.pkl", help="元数据文件路径")
    parser.add_argument("--top-k", type=int, default=None, help="返回前K个结果")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--same-node", action="store_true", help="执行同节点查询（输出该帧所在节点下的所有帧）")
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_mode(
                model_tree_path=args.model_tree,
                model_metadata_path=args.model_metadata,
                top_k=args.top_k,
                verbose=not args.quiet,
            )
            sys.exit(0)

        # 检查输入文件
        if args.skeleton_npz_file:
            if args.frame_index is None:
                print("错误: 必须提供 --frame-index。")
                sys.exit(1)

            if args.same_node:
                results = query_same_node_frames_from_skeleton_npz(
                    npz_file=args.skeleton_npz_file,
                    frame_index=args.frame_index,
                    model_tree_path=args.model_tree,
                    model_metadata_path=args.model_metadata,
                    verbose=not args.quiet
                )
            else:
                results = query_from_skeleton_npz(
                    npz_file=args.skeleton_npz_file,
                    frame_index=args.frame_index,
                    model_tree_path=args.model_tree,
                    model_metadata_path=args.model_metadata,
                    top_k=args.top_k,
                    verbose=not args.quiet
                )
        elif args.npy_file:
            # NPY 模式
            if args.frame_index is None:
                print("错误: 必须提供 --frame-index。")
                sys.exit(1)
            
            if args.same_node:
                results = query_same_node_frames_from_npy(
                    npy_file=args.npy_file,
                    frame_index=args.frame_index,
                    model_tree_path=args.model_tree,
                    model_metadata_path=args.model_metadata,
                    verbose=not args.quiet
                )
            else:
                results = query_from_npy(
                    npy_file=args.npy_file,
                    frame_index=args.frame_index,
                    model_tree_path=args.model_tree,
                    model_metadata_path=args.model_metadata,
                    top_k=args.top_k,
                    verbose=not args.quiet
                )
        elif args.bvh_file:
            # BVH 模式
            if args.frame_index is None:
                print("错误: 必须提供 --frame-index。")
                sys.exit(1)
            
            if args.same_node:
                results = query_same_node_frames_from_bvh(
                    bvh_file=args.bvh_file,
                    frame_index=args.frame_index,
                    model_tree_path=args.model_tree,
                    model_metadata_path=args.model_metadata,
                    verbose=not args.quiet
                )
            else:
                results = query_from_bvh(
                    bvh_file=args.bvh_file,
                    frame_index=args.frame_index,
                    model_tree_path=args.model_tree,
                    model_metadata_path=args.model_metadata,
                    top_k=args.top_k,
                    verbose=not args.quiet
                )
        else:
            print("错误: 必须提供 --bvh-file、--npy-file 或 --skeleton-npz-file。")
            sys.exit(1)
            
        sys.exit(0)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
