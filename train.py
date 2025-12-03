"""
训练脚本：从所有BVH文件加载数据，构建八叉树索引。
"""

from __future__ import annotations

import sys
import time
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from data_loader import load_all_bvh_files, get_bvh_frame_count, load_keypoints_from_bvh
from data_structures import FrameMetadata
from octree_builder import create_root_node, insert_frame, save_tree, save_metadata
import config


def generate_frame_id(bvh_file: str, frame_index: int) -> str:
    """
    生成帧的唯一标识符。
    
    参数:
        bvh_file: BVH文件路径
        frame_index: 帧索引
    
    返回:
        格式如 "01_01_frame_0042" 的唯一标识
    """
    # 从文件路径提取文件名（不含扩展名）
    filename = Path(bvh_file).stem
    # 生成唯一ID，帧号补齐为4位
    frame_id = f"{filename}_frame_{frame_index:04d}"
    return frame_id


def _load_frame_worker(payload: tuple[int, str]) -> dict:
    """子进程：加载单帧关键点并返回结果。"""
    frame_index, bvh_file = payload
    try:
        keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
        frame_id = generate_frame_id(bvh_file, frame_index)
        rounded_keypoints = {
            name: np.round(pos, config.JSON_FLOAT_PRECISION)
            for name, pos in keypoints.items()
        }
        return {
            "success": True,
            "frame_index": frame_index,
            "bvh_file": bvh_file,
            "frame_id": frame_id,
            "keypoints": keypoints,
            "rounded_keypoints": rounded_keypoints,
        }
    except Exception as exc:
        return {
            "success": False,
            "frame_index": frame_index,
            "bvh_file": bvh_file,
            "error": str(exc),
        }


def _normalize_joint_pairs() -> tuple[tuple[str, ...], ...]:
    pairs = getattr(config, "JOINT_PAIR_GROUPS", ())
    normalized = []
    for pair in pairs:
        pair_tuple = tuple(pair)
        if len(pair_tuple) != 2:
            raise ValueError("JOINT_PAIR_GROUPS 中的每个元素必须包含两个关节名称")
        normalized.append(pair_tuple)
    if normalized:
        return tuple(normalized)
    # 如果未配置，退回到单棵树（使用全量关键点）
    return (tuple(config.OCTREE_KEYPOINT_NAMES),)


def _build_tree_base_path(model_tree_path: str) -> tuple[Path, str]:
    path = Path(model_tree_path)
    suffix = path.suffix or ".tree"
    base = path.with_suffix("")
    return base, suffix


def _build_pair_tree_path(base: Path, suffix: str, label: str) -> Path:
    return base.with_name(f"{base.name}_{label}").with_suffix(suffix)


def _build_pair_index_path(base: Path) -> Path:
    return base.parent / f"{base.name}_pairs.json"


def train_model(data_dir: str = "data_train/", 
                model_tree_path: str = "model.tree",
                model_metadata_path: str = "model.pkl",
                verbose: bool = True,
                num_workers: Optional[int] = None) -> None:
    """
    训练帧检索模型。
    
    参数:
        data_dir: 数据目录路径
        model_tree_path: 八叉树模型保存路径（多棵树时作为前缀）
        model_metadata_path: 元数据保存路径
        verbose: 是否打印详细信息
        num_workers: 并行加载BVH的进程数（None表示自动选择）
    """
    if num_workers is None:
        cpu_total = os.cpu_count() or 1
        num_workers = max(1, cpu_total - 1)
    elif num_workers <= 0:
        num_workers = 1

    if verbose:
        print("=" * 80)
        print("开始训练帧检索模型")
        print("=" * 80)
        print(f"并行加载进程数: {num_workers}")
    
    # 1. 扫描所有BVH文件
    if verbose:
        print(f"\n步骤1: 扫描 {data_dir} 目录...")
    
    bvh_files = load_all_bvh_files(data_dir)
    if verbose:
        print(f"找到 {len(bvh_files)} 个BVH文件:")
        for bvh_file in bvh_files:
            print(f"  - {Path(bvh_file).name}")
    
    # 2. 创建八叉树根节点（每个关节对一棵树）
    joint_pairs = _normalize_joint_pairs()
    use_multi_tree = len(joint_pairs) > 1 or joint_pairs[0] != tuple(config.OCTREE_KEYPOINT_NAMES)
    pair_trees: dict[str, dict] = {}

    if verbose:
        print("\n步骤2: 创建八叉树根节点...")
        if use_multi_tree:
            print(f"启用多棵树模式，共 {len(joint_pairs)} 个关节对。")
        else:
            print("使用单棵树（全量关键点）模式。")
    
    for pair in joint_pairs:
        label = "_".join(pair)
        root = create_root_node(pair)
        pair_trees[label] = {
            "keypoints": pair,
            "root": root,
        }
        if verbose:
            print(f"  - 关节对 {label} -> 八叉树关键点: {', '.join(pair)}")
    
    # 3. 遍历所有文件和帧，插入到每棵八叉树
    if verbose:
        print("\n步骤3: 加载并插入所有帧...")
    
    metadata_list: List[FrameMetadata] = []
    total_frames = 0
    error_count = 0
    
    total_start_time = time.time()
    
    executor: ProcessPoolExecutor | None = None
    if num_workers > 1:
        executor = ProcessPoolExecutor(max_workers=num_workers)

    try:
        for bvh_file in bvh_files:
            try:
                frame_count = get_bvh_frame_count(bvh_file)
                if verbose:
                    print(f"\n处理文件: {Path(bvh_file).name} ({frame_count} 帧)")
                
                file_start_time = time.time()
                file_frame_count = 0
                
                if executor:
                    futures = [
                        executor.submit(_load_frame_worker, (frame_index, bvh_file))
                        for frame_index in range(frame_count)
                    ]
                    results_iter = (future.result() for future in as_completed(futures))
                else:
                    results_iter = (_load_frame_worker((frame_index, bvh_file)) for frame_index in range(frame_count))

                for result in results_iter:
                    if result["success"]:
                        keypoints = result["keypoints"]
                        frame_id = result["frame_id"]

                        for pair_data in pair_trees.values():
                            insert_frame(pair_data["root"], keypoints, frame_id)

                        metadata = FrameMetadata(
                            bvh_file=result["bvh_file"],
                            frame_index=result["frame_index"],
                            frame_id=frame_id,
                            keypoints=result["rounded_keypoints"],
                        )
                        metadata_list.append(metadata)
                        
                        total_frames += 1
                        file_frame_count += 1
                        
                        if verbose and total_frames % 100 == 0:
                            print(f"  已处理 {total_frames} 帧...", end='\r')
                    else:
                        error_count += 1
                        if verbose:
                            print(f"  警告: 无法加载帧 {result['frame_index']}: {result['error']}")
                
                file_elapsed = time.time() - file_start_time
                total_elapsed = time.time() - total_start_time
                
                if verbose:
                    avg = file_elapsed / file_frame_count if file_frame_count else 0
                    print(f"  ✓ 完成 {Path(bvh_file).name}: {file_frame_count} 帧")
                    print(f"    文件用时: {file_elapsed:.2f} 秒 (平均: {avg:.4f} 秒/帧)")
                    print(f"    总时长: {total_elapsed:.2f} 秒")
            
            except Exception as e:
                if verbose:
                    print(f"  错误: 无法处理文件 {Path(bvh_file).name}: {e}")
    finally:
        if executor:
            executor.shutdown(wait=True)
    
    if verbose:
        print(f"\n\n训练完成!")
        print(f"  成功加载帧数: {total_frames}")
        print(f"  错误帧数: {error_count}")
    
    # 4. 保存模型
    if verbose:
        print("\n步骤4: 保存模型...")
    
    base_path, suffix = _build_tree_base_path(model_tree_path)
    pair_index_entries = []
    total_tree_size = 0.0

    for label, pair_data in pair_trees.items():
        tree_path = _build_pair_tree_path(base_path, suffix, label)
        save_tree(pair_data["root"], str(tree_path), show_progress=verbose)
        size_mb = Path(tree_path).stat().st_size / 1024 / 1024
        total_tree_size += size_mb
        pair_index_entries.append({
            "label": label,
            "keypoints": list(pair_data["keypoints"]),
            "tree_file": str(tree_path),
        })
        if verbose:
            print(f"  八叉树[{label}] -> {tree_path} ({size_mb:.2f} MB)")

    pair_index_path = _build_pair_index_path(base_path)
    with open(pair_index_path, "w", encoding="utf-8") as index_file:
        json.dump({"pairs": pair_index_entries}, index_file, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  关节对索引已保存到: {pair_index_path}")

    save_metadata(metadata_list, model_metadata_path)
    metadata_size = Path(model_metadata_path).stat().st_size / 1024 / 1024
    if verbose:
        print(f"  元数据已保存到: {model_metadata_path} ({metadata_size:.2f} MB)")
        print(f"  树文件总大小: {total_tree_size:.2f} MB")
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
def count_nodes(node) -> int:
    """递归统计节点数。"""
    count = 1
    for _, child in node.iter_children():
        count += count_nodes(child)
    return count


def count_leaf_nodes(node) -> int:
    """递归统计叶节点数。"""
    if node.is_leaf():
        return 1
    count = 0
    for _, child in node.iter_children():
        count += count_leaf_nodes(child)
    return count


def get_max_frames_in_leaf(node) -> int:
    """获取单个叶节点中的最大帧数。"""
    if node.is_leaf():
        return len(node.frame_ids)
    max_frames = 0
    for _, child in node.iter_children():
        max_frames = max(max_frames, get_max_frames_in_leaf(child))
    return max_frames


def main():
    """主函数。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练帧检索模型")
    parser.add_argument("--data-dir", default="data_train/", help="数据目录路径")
    parser.add_argument("--output-tree", default="model.tree", help="输出树文件路径")
    parser.add_argument("--output-metadata", default="model.pkl", help="输出元数据文件路径")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不打印详细信息")
    parser.add_argument("--workers", type=int, default=None, help="并行加载进程数（默认CPU核心数-1）")
    
    args = parser.parse_args()
    
    try:
        train_model(
            data_dir=args.data_dir,
            model_tree_path=args.output_tree,
            model_metadata_path=args.output_metadata,
            verbose=not args.quiet,
            num_workers=args.workers,
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
