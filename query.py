"""
帧检索查询脚本：加载模型并执行相似帧查询。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

from data_loader import load_keypoints_from_bvh
from octree_builder import load_tree, load_metadata
from similarity import find_similar_frames, SimilarityResult
import config


def query_frame(query_keypoints: dict[str, np.ndarray],
               model_tree_path: str = "model.tree",
               model_metadata_path: str = "model.pkl",
               top_k: int = None,
               verbose: bool = True) -> List[SimilarityResult]:
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
    
    # 开始计时 - 加载模型
    load_start_time = time.time()
    
    tree = load_tree(model_tree_path)
    metadata_list = load_metadata(model_metadata_path)
    
    load_elapsed = time.time() - load_start_time
    
    if verbose:
        print(f"  成功加载八叉树模型: {model_tree_path}")
        print(f"  成功加载元数据: {model_metadata_path}")
        print(f"  训练集帧数: {len(metadata_list)}")
        print(f"  加载模型用时: {load_elapsed:.4f} 秒")
    
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
    
    # 3. 查找相似帧
    if verbose:
        print(f"\n步骤3: 查找Top-{top_k}相似帧...")
    
    # 开始计时 - 查询
    query_start_time = time.time()
    
    results = find_similar_frames(query_keypoints, metadata_list, top_k)
    
    query_elapsed = time.time() - query_start_time
    
    if verbose:
        print(f"  查询用时: {query_elapsed:.4f} 秒")
    
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
        
        print("\n说明:")
        print("  - 距离值: 加权欧氏距离，越小表示越相似")
        print("  - 相似度: 0-1之间，1表示完全相同，0表示完全不同")
        print(f"  - 精确匹配阈值: 距离 < {config.EXACT_MATCH_EPSILON}")
        print("\n性能:")
        print(f"  - 加载模型用时: {load_elapsed:.4f} 秒")
        print(f"  - 查询用时: {query_elapsed:.4f} 秒")
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


def query_from_bvh(bvh_file: str,
                  frame_index: int,
                  model_tree_path: str = "model.tree",
                  model_metadata_path: str = "model.pkl",
                  top_k: int = None,
                  verbose: bool = True) -> List[SimilarityResult]:
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
    return query_frame(keypoints, model_tree_path, model_metadata_path, top_k, verbose)


def main():
    """主函数。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="帧检索查询")
    parser.add_argument("--bvh-file", required=True, help="BVH文件路径")
    parser.add_argument("--frame-index", type=int, required=True, help="帧索引")
    parser.add_argument("--model-tree", default="model.tree", help="树模型文件路径")
    parser.add_argument("--model-metadata", default="model.pkl", help="元数据文件路径")
    parser.add_argument("--top-k", type=int, default=None, help=f"返回前K个结果（默认{config.TOP_K}）")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    try:
        # 检查文件是否存在
        if not Path(args.bvh_file).exists():
            print(f"错误: BVH文件不存在: {args.bvh_file}")
            sys.exit(1)
        
        if not Path(args.model_tree).exists():
            print(f"错误: 树模型文件不存在: {args.model_tree}")
            print("请先运行 train.py 训练模型")
            sys.exit(1)
        
        if not Path(args.model_metadata).exists():
            print(f"错误: 元数据文件不存在: {args.model_metadata}")
            print("请先运行 train.py 训练模型")
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
