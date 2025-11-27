"""
帧检索查询脚本：加载模型并执行相似帧查询。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from data_loader import load_keypoints_from_bvh
from octree_builder import load_tree, load_metadata
from similarity import find_similar_frames_in_candidates, SimilarityResult
from data_structures import coerce_body_keypoints, compute_octant, FrameMetadata
from octree_node import ActionTreeNode
import config


_MODEL_CACHE: dict[Tuple[str, str], Tuple[ActionTreeNode, List[FrameMetadata]]] = {}


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
    
    # 1. 在八叉树中向下遍历到叶节点
    current = tree
    
    for depth in range(config.MAX_DEPTH):
        # 只为用于八叉树的关键点计算octant（不包括hip原点）
        octants = tuple(
            compute_octant(keypoint_dict[name], current.bboxes[name])
            for name in config.OCTREE_KEYPOINT_NAMES
        )
        
        # 尝试获取子节点
        child = current.get_child(octants)
        if child is None:
            # 没有子节点，停在当前节点
            break
        current = child
    
    # 2. 获取当前节点的候选帧
    candidate_frame_ids = list(current.get_frame_ids())
    
    # 3. 智能回溯：如果候选数量不足，向上回溯父节点扩充候选集
    visited_nodes = {id(current)}  # 避免重复添加
    
    while len(candidate_frame_ids) < min_candidates and current.parent is not None:
        current = current.parent
        node_id = id(current)
        
        if node_id not in visited_nodes:
            # 添加父节点的帧ID（去重）
            for frame_id in current.get_frame_ids():
                if frame_id not in candidate_frame_ids:
                    candidate_frame_ids.append(frame_id)
            visited_nodes.add(node_id)
    
    return candidate_frame_ids


def _load_model_once(model_tree_path: str,
                     model_metadata_path: str,
                     use_cache: bool = True) -> tuple[ActionTreeNode, List[FrameMetadata], bool]:
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

    tree = load_tree(model_tree_path)
    metadata_list = load_metadata(model_metadata_path)

    if use_cache:
        _MODEL_CACHE[cache_key] = (tree, metadata_list)

    return tree, metadata_list, False


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
    执行帧检索查询。
    
    参数:
        query_keypoints: 查询帧的关键点坐标
        model_tree_path: 树模型文件路径
        model_metadata_path: 元数据文件路径
        top_k: 返回前K个最相似的帧（默认使用config.TOP_K）
        verbose: 是否打印详细信息
    
    返回:
        相似度结果列表
    """
    if top_k is None:
        top_k = config.TOP_K
    
    # 1. 加载模型
    if verbose:
        print("=" * 80)
        print("帧检索查询系统")
        print("=" * 80)
        print("\n步骤1: 加载模型...")
    
    load_elapsed = 0.0
    if tree_instance is not None and metadata_instance is not None:
        tree = tree_instance
        metadata_list = metadata_instance
        from_cache = True
        if verbose:
            print("  使用传入的模型实例，无需重新加载。")
    else:
        load_start_time = time.time()
        tree, metadata_list, from_cache = _load_model_once(
            model_tree_path,
            model_metadata_path,
            use_cache=use_cache,
        )
        if from_cache:
            if verbose:
                print("  使用内存缓存的模型，无需再次读取磁盘。")
        else:
            load_elapsed = time.time() - load_start_time
            if verbose:
                print(f"  成功加载八叉树模型: {model_tree_path}")
                print(f"  成功加载元数据: {model_metadata_path}")
                print(f"  加载模型用时: {load_elapsed:.4f} 秒")
    
    if verbose:
        print(f"  训练集帧数: {len(metadata_list)}")
    
    # 2. 打印查询帧的关键点信息
    if verbose:
        print("\n步骤2: 查询帧关键点位置...")
        print("-" * 80)
        for joint_name in config.KEYPOINT_NAMES:
            if joint_name in query_keypoints:
                pos = query_keypoints[joint_name]
                print(f"  {joint_name:12s}: X={pos[0]:8.3f}, Y={pos[1]:8.3f}, Z={pos[2]:8.3f}")
            else:
                print(f"  {joint_name:12s}: <缺失>")
        print("-" * 80)
    
    # 3. 使用八叉树查找候选帧
    if verbose:
        print(f"\n步骤3: 使用八叉树空间索引筛选候选帧...")
    
    # 开始计时 - 八叉树定位
    tree_search_start_time = time.time()
    
    candidate_frame_ids = find_candidate_frames_from_tree(tree, query_keypoints)
    
    tree_search_elapsed = time.time() - tree_search_start_time
    
    if verbose:
        print(f"  八叉树定位用时: {tree_search_elapsed:.4f} 秒")
        print(f"  候选帧数量: {len(candidate_frame_ids)}/{len(metadata_list)}")
        print(f"  筛选比例: {len(candidate_frame_ids)/len(metadata_list)*100:.2f}%")
    
    # 4. 在候选帧中查找Top-K相似帧
    if verbose:
        print(f"\n步骤4: 在候选帧中计算精确距离并查找Top-{top_k}相似帧...")
    
    # 开始计时 - 精确计算
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
    
    # 总查询时间
    query_elapsed = tree_search_elapsed + similarity_elapsed
    
    if verbose:
        print(f"  精确计算用时: {similarity_elapsed:.4f} 秒")
        print(f"  总查询用时: {query_elapsed:.4f} 秒")
    
    # 4. 打印结果
    if verbose:
        print("\n查询结果:")
        print("=" * 80)
        
        # 检查是否有精确匹配
        exact_matches = [r for r in results if r.is_exact_match]
        if exact_matches:
            print("\n[精确匹配] 找到完全相同的帧！")
            print("-" * 80)
            for i, result in enumerate(exact_matches, 1):
                print_result(result, i)
            print("-" * 80)
        else:
            print("\n未找到精确匹配，以下是最相似的帧：")
        
        # 打印Top-K结果
        print(f"\nTop-{top_k}最相似的帧:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print_result(result, i)
        print("-" * 80)
        
        # 打印排名第一的帧的详细坐标对比
        if results:
            print("\n" + "=" * 80)
            print("排名第一的帧详细坐标对比:")
            print("=" * 80)
            print_coordinate_comparison(query_keypoints, results[0])
        
        print("\n说明:")
        print("  - 距离值: 加权欧氏距离，越小表示越相似")
        print("  - 相似度: 0-1之间，1表示完全相同，0表示完全不同")
        print(f"  - 精确匹配阈值: 距离 < {config.EXACT_MATCH_EPSILON}")
        print("\n性能统计:")
        print(f"  - 加载模型用时: {load_elapsed:.4f} 秒")
        print(f"  - 八叉树定位用时: {tree_search_elapsed:.4f} 秒")
        print(f"  - 精确计算用时: {similarity_elapsed:.4f} 秒")
        print(f"  - 总查询用时: {query_elapsed:.4f} 秒")
        print(f"  - 候选帧数量: {len(candidate_frame_ids)}/{len(metadata_list)}")
        print(f"  - 筛选比例: {len(candidate_frame_ids)/len(metadata_list)*100:.2f}%")
        print(f"  - 总用时: {load_elapsed + query_elapsed:.4f} 秒")
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
