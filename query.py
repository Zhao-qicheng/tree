"""
帧检索查询脚本：加载模型并执行相似帧查询。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
from collections import Counter

import numpy as np

from data_loader import load_keypoints_from_bvh
from octree_builder import load_tree, load_metadata
from similarity import find_similar_frames_in_candidates, SimilarityResult
from data_structures import coerce_body_keypoints, compute_octant, FrameMetadata, BoundingBox
from octree_node import ActionTreeNode
from rotation_utils import create_custom_rotation_configs, RotationConfig
import config


_MODEL_CACHE: dict[Tuple[str, str], Tuple[ActionTreeNode, List[FrameMetadata]]] = {}


def _estimate_node_distance(node: ActionTreeNode,
                            keypoint_dict: dict[str, np.ndarray]) -> float:
    """
    根据节点包围盒中心与查询关键点的距离估算“接近程度”。
    距离越小，表示节点越可能包含相似帧。
    """
    total = 0.0
    count = 0
    for name in config.OCTREE_KEYPOINT_NAMES:
        if name not in keypoint_dict:
            continue
        bbox: BoundingBox | None = node.bboxes.get(name)
        if bbox is None:
            continue
        center = bbox.center()
        total += float(np.linalg.norm(keypoint_dict[name] - center))
        count += 1
    if count == 0:
        return float("inf")
    return total / count


def find_candidate_frames_from_tree(tree: ActionTreeNode,
                                    query_keypoints: dict[str, np.ndarray],
                                    min_candidates: int = None) -> list[str]:
    """
    使用八叉树空间索引查找候选帧。
    
    通过在八叉树中向下遍历，定位到查询帧所在的叶节点，获取候选帧ID列表。
    如果候选数量不足，向上回溯父节点扩充候选集。
    
    参数:
        tree: 八叉树根节点
        query_keypoints: 查询帧的关键点坐标
        min_candidates: 最小候选帧数量（默认使用config.MIN_CANDIDATES）
    
    返回:
        候选帧ID列表
    """
    if min_candidates is None:
        min_candidates = config.MIN_CANDIDATES
    
    # 标准化查询关键点
    body = coerce_body_keypoints(query_keypoints)
    keypoint_dict = body.as_dict()
    
    # 1. 使用 Beam Search 在八叉树中向下遍历，避免单一路径失效
    beam_width = max(1, getattr(config, "BEAM_WIDTH", 4))
    beam_nodes: list[tuple[ActionTreeNode, float]] = [(tree, 0.0)]
    leaf_nodes: list[ActionTreeNode] = []
    
    for _ in range(config.MAX_DEPTH):
        next_candidates: list[tuple[float, ActionTreeNode]] = []
        for node, _score in beam_nodes:
            if not node.children:
                leaf_nodes.append(node)
                continue
            for _, child in node.iter_children():
                score = _estimate_node_distance(child, keypoint_dict)
                next_candidates.append((score, child))
        if not next_candidates:
            break
        next_candidates.sort(key=lambda item: item[0])
        beam_nodes = [(child, score) for score, child in next_candidates[:beam_width]]
    
    candidate_nodes: list[ActionTreeNode] = [node for node, _ in beam_nodes]
    candidate_nodes.extend(leaf_nodes)
    
    candidate_frame_ids: list[str] = []
    seen_frame_ids: set[str] = set()
    visited_nodes: set[int] = set()
    
    for node in candidate_nodes:
        if node is None:
            continue
        visited_nodes.add(id(node))
        for frame_id in node.get_frame_ids():
            if frame_id not in seen_frame_ids:
                candidate_frame_ids.append(frame_id)
                seen_frame_ids.add(frame_id)
    
    # 2. 控制回溯层级：仅在必要时回溯少量层级，避免候选集膨胀
    current_layer = candidate_nodes
    backtrack_depth = 0
    max_backtrack_depth = getattr(config, "MAX_BACKTRACK_DEPTH", 2)
    
    while (len(candidate_frame_ids) < min_candidates and
           current_layer and
           backtrack_depth < max_backtrack_depth):
        parents: list[ActionTreeNode] = []
        for node in current_layer:
            parent = getattr(node, "parent", None)
            if parent is None or id(parent) in visited_nodes:
                continue
            visited_nodes.add(id(parent))
            parents.append(parent)
            for frame_id in parent.get_frame_ids():
                if frame_id not in seen_frame_ids:
                    candidate_frame_ids.append(frame_id)
                    seen_frame_ids.add(frame_id)
        current_layer = parents
        backtrack_depth += 1
    
    return candidate_frame_ids


def merge_candidates(all_candidates: Dict[int, List[str]],
                    strategy: str = None,
                    min_vote_threshold: int = None,
                    verbose: bool = False) -> List[str]:
    """
    合并多棵树的候选帧。
    
    参数:
        all_candidates: {tree_id: [frame_ids]}
        strategy: 合并策略 ("vote", "union", "intersection")
        min_vote_threshold: 最小投票数（仅用于vote策略）
        verbose: 是否打印详细信息
    
    返回:
        合并后的候选帧ID列表
    """
    if strategy is None:
        strategy = config.MERGE_STRATEGY
    if min_vote_threshold is None:
        min_vote_threshold = config.MIN_VOTE_THRESHOLD
    
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


def _load_model_once(model_tree_path: str,
                     model_metadata_path: str,
                     use_cache: bool = True,
                     show_progress: bool = True) -> tuple[ActionTreeNode, List[FrameMetadata], bool]:
    """
    加载模型，如果启用缓存则返回内存中已有的模型。

    返回:
        (tree, metadata_list, from_cache)
    """
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
                     tree_instance: Optional[ActionTreeNode] = None,
                     metadata_instance: Optional[List[FrameMetadata]] = None,
                     use_cache: bool = True,
                     enable_parallel: bool = True,
                     parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """
    在单树中查询相似帧。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        model_tree_path: 树模型文件路径
        model_metadata_path: 元数据文件路径
        rotation_config: 旋转配置（如果是旋转树）
        top_k: 返回前K个最相似的帧
        verbose: 是否打印详细信息
        其他参数同query_frame
    """
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
    
    # 2. 应用旋转（如果需要）
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
        query_keypoints,  # 使用原始坐标计算距离
        metadata_list,
        candidate_frame_ids,
        top_k,
        enable_parallel=enable_parallel,
        max_workers=parallel_workers,
    )
    similarity_elapsed = time.time() - similarity_start_time
    
    return results, candidate_frame_ids, load_elapsed, tree_search_elapsed, similarity_elapsed


def query_frame(query_keypoints: dict[str, np.ndarray],
               model_tree_path: str = "model.tree",
               model_metadata_path: str = "model.pkl",
               top_k: int = None,
               verbose: bool = True,
               *,
               tree_instance: Optional[ActionTreeNode] = None,
               metadata_instance: Optional[List[FrameMetadata]] = None,
               use_cache: bool = True,
               enable_parallel: bool = True,
               parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """
    执行帧检索查询（支持单树和多树模式）。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        model_tree_path: 树模型文件路径（单树）或基础路径（多树）
        model_metadata_path: 元数据文件路径（单树）或基础路径（多树）
        top_k: 返回前K个最相似的帧（默认使用config.TOP_K）
        verbose: 是否打印详细信息
    
    返回:
        相似度结果列表
    """
    if top_k is None:
        top_k = config.TOP_K
    
    # 检查是否启用多树模式
    if not config.ENABLE_MULTI_TREE:
        # 单树模式
        results, candidate_ids, load_elapsed, tree_elapsed, sim_elapsed = query_single_tree(
            query_keypoints=query_keypoints,
            model_tree_path=model_tree_path,
            model_metadata_path=model_metadata_path,
            rotation_config=None,
            top_k=top_k,
            verbose=False,
            tree_instance=tree_instance,
            metadata_instance=metadata_instance,
            use_cache=use_cache,
            enable_parallel=enable_parallel,
            parallel_workers=parallel_workers,
        )
        
        # 打印详细信息（单树模式）
        if verbose:
            print("=" * 80)
            print("帧检索查询系统（单树模式）")
            print("=" * 80)
            print(f"\n查询完成！")
            print(f"  候选帧数量: {len(candidate_ids)}")
            print(f"  八叉树查询用时: {tree_elapsed:.4f} 秒")
            print(f"  精确计算用时: {sim_elapsed:.4f} 秒")
            print(f"  总查询用时: {tree_elapsed + sim_elapsed:.4f} 秒\n")
            
            # 打印Top-K结果
            print(f"Top-{top_k}最相似的帧:")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                print_result(result, i)
            print("-" * 80)
        
        return results
    
    # 多树模式
    num_trees = len(config.ROTATION_CONFIGS)
    
    if verbose:
        print("=" * 80)
        print(f"帧检索查询系统（多树模式 - {num_trees}棵树）")
        print("=" * 80)
    
    # 创建旋转配置对象
    rotation_configs = create_custom_rotation_configs(config.ROTATION_CONFIGS)
    
    # 生成模型路径
    base_tree_path = Path(model_tree_path).stem
    base_metadata_path = Path(model_metadata_path).stem
    model_dir = Path(model_tree_path).parent
    
    # 1. 加载所有树（只加载一次元数据）
    if verbose:
        print(f"\n步骤1: 加载 {num_trees} 棵树...")
    
    trees: Dict[int, ActionTreeNode] = {}
    metadata_list: List[FrameMetadata] = None
    load_start = time.time()
    
    for rot_config in rotation_configs:
        tree_path = str(model_dir / rot_config.get_model_filename(base_tree_path))
        metadata_path = str(model_dir / rot_config.get_metadata_filename(base_metadata_path))
        
        trees[rot_config.tree_id] = load_tree(tree_path, show_progress=False)
        
        # 只加载一次元数据（使用树0的元数据，包含原始坐标）
        if metadata_list is None and rot_config.tree_id == 0:
            metadata_list = load_metadata(metadata_path)
    
    load_elapsed = time.time() - load_start
    
    if verbose:
        print(f"  已加载 {num_trees} 棵树")
        print(f"  训练集帧数: {len(metadata_list)}")
        print(f"  加载用时: {load_elapsed:.4f} 秒\n")
    
    # 2. 对每棵树进行查询
    if verbose:
        print(f"步骤2: 从每棵树查找候选帧...")
    
    all_candidates: Dict[int, List[str]] = {}
    query_start = time.time()
    
    for rot_config in rotation_configs:
        # 旋转查询关键点
        rotated_query = rot_config.rotate(query_keypoints)
        
        # 在该树中查找候选帧
        candidates = find_candidate_frames_from_tree(
            trees[rot_config.tree_id],
            rotated_query,
            min_candidates=config.MIN_CANDIDATES
        )
        
        all_candidates[rot_config.tree_id] = candidates
        
        if verbose:
            print(f"  树{rot_config.tree_id}: {len(candidates)} 个候选帧")
    
    # 3. 合并候选帧
    if verbose:
        print(f"\n步骤3: 合并候选帧...")
    
    merged_candidates = merge_candidates(
        all_candidates,
        strategy=config.MERGE_STRATEGY,
        min_vote_threshold=config.MIN_VOTE_THRESHOLD,
        verbose=verbose
    )
    
    tree_elapsed = time.time() - query_start
    
    if verbose:
        print(f"  合并后候选帧数: {len(merged_candidates)}")
        print(f"  八叉树查询用时: {tree_elapsed:.4f} 秒\n")
    
    # 4. 在合并后的候选集中计算精确距离
    if verbose:
        print(f"步骤4: 计算精确距离并查找Top-{top_k}...\n")
    
    sim_start = time.time()
    results = find_similar_frames_in_candidates(
        query_keypoints,
        metadata_list,
        merged_candidates,
        top_k,
        enable_parallel=enable_parallel,
        max_workers=parallel_workers,
    )
    sim_elapsed = time.time() - sim_start
    
    total_query_time = tree_elapsed + sim_elapsed
    
    if verbose:
        print(f"  精确计算用时: {sim_elapsed:.4f} 秒")
        print(f"  总查询用时: {total_query_time:.4f} 秒\n")
        
        # 打印Top-K结果
        print("=" * 80)
        print(f"Top-{top_k}最相似的帧:")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            print_result(result, i)
        print("-" * 80)
        
        print(f"\n性能统计:")
        print(f"  - 加载模型用时: {load_elapsed:.4f} 秒")
        print(f"  - 八叉树查询用时: {tree_elapsed:.4f} 秒")
        print(f"  - 精确计算用时: {sim_elapsed:.4f} 秒")
        print(f"  - 总用时: {load_elapsed + total_query_time:.4f} 秒")
        print("=" * 80)
    
    return results


def print_result(result: SimilarityResult, rank: int) -> None:
    """
    打印单个查询结果。
    
    参数:
        result: 相似度结果
        rank: 排名
    """
    metadata = result.frame_metadata
    filename = Path(metadata.bvh_file).name
    
    # 格式化输出
    match_flag = "[精确匹配]" if result.is_exact_match else ""
    
    print(f"\n排名 {rank}: {match_flag}")
    print(f"  文件名: {filename}")
    print(f"  帧索引: {metadata.frame_index}")
    print(f"  帧ID: {metadata.frame_id}")
    print(f"  距离值: {result.distance:.6f}")
    print(f"  相似度: {result.similarity_score:.4f}")


def print_coordinate_comparison(query_keypoints: dict[str, np.ndarray], 
                               best_match: SimilarityResult) -> None:
    """
    打印查询帧与最佳匹配帧的关节坐标对比。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        best_match: 最佳匹配结果
    """
    matched_keypoints = best_match.frame_metadata.keypoints
    
    print("\n关节坐标对比 (查询帧 vs 匹配帧):")
    print("-" * 120)
    print(f"{'关节名称':<12} | {'查询帧 X':>12} {'Y':>12} {'Z':>12} | {'匹配帧 X':>12} {'Y':>12} {'Z':>12} | {'差值':>12}")
    print("-" * 120)
    
    total_diff = 0.0
    for joint_name in config.KEYPOINT_NAMES:
        if joint_name in query_keypoints and joint_name in matched_keypoints:
            query_pos = query_keypoints[joint_name]
            match_pos = matched_keypoints[joint_name]
            
            # 计算欧氏距离
            diff = np.linalg.norm(query_pos - match_pos)
            total_diff += diff
            
            # 格式化输出
            print(f"{joint_name:<12} | "
                  f"{query_pos[0]:12.6f} {query_pos[1]:12.6f} {query_pos[2]:12.6f} | "
                  f"{match_pos[0]:12.6f} {match_pos[1]:12.6f} {match_pos[2]:12.6f} | "
                  f"{diff:12.6f}")
        else:
            print(f"{joint_name:<12} | {'<缺失>':>38} | {'<缺失>':>38} | {'N/A':>12}")
    
    print("-" * 120)
    print(f"{'总欧氏距离':>50}: {total_diff:12.6f}")
    print(f"{'加权距离(相似度计算用)':>50}: {best_match.distance:12.6f}")
    
    # 如果是精确匹配，特别标注
    if best_match.is_exact_match:
        print(f"\n✓ 该帧为精确匹配 (加权距离 < {config.EXACT_MATCH_EPSILON})")
    else:
        print(f"\n✗ 该帧不是精确匹配 (加权距离 >= {config.EXACT_MATCH_EPSILON})")
    
    print("=" * 120)


def query_from_bvh(bvh_file: str,
                  frame_index: int,
                  model_tree_path: str = "model.tree",
                  model_metadata_path: str = "model.pkl",
                  top_k: int = None,
                  verbose: bool = True,
                  *,
                  tree_instance: Optional[ActionTreeNode] = None,
                  metadata_instance: Optional[List[FrameMetadata]] = None,
                  use_cache: bool = True,
                  enable_parallel: bool = True,
                  parallel_workers: Optional[int] = None) -> List[SimilarityResult]:
    """
    从BVH文件加载指定帧并执行查询。
    
    参数:
        bvh_file: BVH文件路径
        frame_index: 帧索引
        model_tree_path: 树模型文件路径
        model_metadata_path: 元数据文件路径
        top_k: 返回前K个最相似的帧
        verbose: 是否打印详细信息
    
    返回:
        相似度结果列表
    """
    if verbose:
        print(f"\n从BVH文件加载查询帧...")
        print(f"  文件: {bvh_file}")
        print(f"  帧索引: {frame_index}")
    
    # 加载关键点
    keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
    
    # 执行查询
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
    """交互式查询模式：加载一次模型，多次执行查询。"""
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
                print("\n检测到输入流结束，退出交互模式。")
                break
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出交互模式。")
            break

        if not user_input:
            continue

        if user_input.lower() in {"q", "quit", "exit"}:
            print("已退出交互模式。")
            break

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
    """主函数。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="帧检索查询")
    parser.add_argument("--bvh-file", help="BVH文件路径（交互模式可省略）")
    parser.add_argument("--frame-index", type=int, help="帧索引（交互模式可省略）")
    parser.add_argument("--model-tree", default="model.tree", help="树模型文件路径")
    parser.add_argument("--model-metadata", default="model.pkl", help="元数据文件路径")
    parser.add_argument("--top-k", type=int, default=None, help=f"返回前K个结果（默认{config.TOP_K}）")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--interactive", action="store_true", help="交互式模式：加载一次模型，多次查询")
    
    args = parser.parse_args()
    
    try:
        if not Path(args.model_tree).exists():
            print(f"错误: 树模型文件不存在: {args.model_tree}")
            print("请先运行 train.py 训练模型")
            sys.exit(1)
        
        if not Path(args.model_metadata).exists():
            print(f"错误: 元数据文件不存在: {args.model_metadata}")
            print("请先运行 train.py 训练模型")
            sys.exit(1)

        if args.interactive:
            interactive_mode(
                model_tree_path=args.model_tree,
                model_metadata_path=args.model_metadata,
                top_k=args.top_k,
                verbose=not args.quiet,
            )
            sys.exit(0)

        if not args.bvh_file or args.frame_index is None:
            print("错误: 非交互模式下必须提供 --bvh-file 与 --frame-index。")
            sys.exit(1)

        if not Path(args.bvh_file).exists():
            print(f"错误: BVH文件不存在: {args.bvh_file}")
            sys.exit(1)
        
        # 执行查询
        results = query_from_bvh(
            bvh_file=args.bvh_file,
            frame_index=args.frame_index,
            model_tree_path=args.model_tree,
            model_metadata_path=args.model_metadata,
            top_k=args.top_k,
            verbose=not args.quiet
        )
        
        # 如果有精确匹配，返回0；否则返回1
        has_exact_match = any(r.is_exact_match for r in results)
        sys.exit(0 if has_exact_match else 1)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
