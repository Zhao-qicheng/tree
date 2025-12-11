"""
训练脚本：从所有BVH文件加载数据，构建八叉树索引（扁平化优化版）。
"""

from __future__ import annotations

import sys
import time
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np

from data_loader import load_all_bvh_files, get_bvh_frame_count, load_keypoints_from_bvh
from data_frame import clear_specific_file
from data_structures import FrameMetadata
from octree_builder import create_root_node, insert_frame, save_tree, save_metadata
from rotation_utils import create_custom_rotation_configs, RotationConfig
import config


def generate_frame_id(bvh_file: str, frame_index: int) -> str:
    filename = Path(bvh_file).stem
    frame_id = f"{filename}_frame_{frame_index:04d}"
    return frame_id


def _load_frame_worker(payload: tuple[int, str]) -> dict:
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


def train_single_tree(data_dir: str,
                      model_tree_path: str,
                      model_metadata_path: str,
                      rotation_config: Optional[RotationConfig] = None,
                      verbose: bool = True,
                      num_workers: Optional[int] = None) -> None:
    if num_workers is None:
        cpu_total = os.cpu_count() or 1
        num_workers = max(1, cpu_total - 1)
    elif num_workers <= 0:
        num_workers = 1

    if verbose:
        print("=" * 80)
        if rotation_config:
            print(f"训练树 {rotation_config.tree_id}: {rotation_config.axis}轴 {rotation_config.angle}°")
        else:
            print("开始训练帧检索模型（单树模式 - 扁平化优化）")
        print("=" * 80)
        print(f"并行加载进程数: {num_workers}")
    
    # 1. 扫描所有BVH文件
    if verbose:
        print(f"\n步骤1: 扫描 {data_dir} 目录...")
    
    bvh_files = load_all_bvh_files(data_dir)
    if verbose:
        print(f"找到 {len(bvh_files)} 个BVH文件")
    
    # 2. 创建八叉树根节点
    if verbose:
        print("\n步骤2: 创建八叉树根节点...")
    
    root = create_root_node()
    if verbose:
        print(f"根节点创建成功，深度: {root.depth}")
    
    # 3. 遍历所有文件和帧，插入到八叉树
    if verbose:
        print("\n步骤3: 加载并插入所有帧（包含旋转增强）...")
    
    metadata_list: List[FrameMetadata] = []
    total_frames = 0
    error_count = 0
    
    # 准备旋转配置
    rot_configs = create_custom_rotation_configs(config.ROTATION_CONFIGS)
    if verbose:
        print(f"  应用 {len(rot_configs)} 种旋转配置进行数据增强")
    
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
                        base_keypoints = result["keypoints"]
                        base_frame_id = result["frame_id"]
                        
                        # 遍历所有旋转配置进行增强
                        for rot_cfg in rot_configs:
                            # 1. 旋转关键点
                            # 如果是0度（原始），rotate方法会直接返回副本或原样，效率较高
                            aug_keypoints = rot_cfg.rotate(base_keypoints)
                            
                            # 2. 生成增强后的 Frame ID
                            # 对于原始角度(0度)，保持原ID，方便辨识
                            if rot_cfg.angle == 0:
                                aug_frame_id = base_frame_id
                            else:
                                aug_frame_id = f"{base_frame_id}_rot_{rot_cfg.axis}{rot_cfg.angle}"
                            
                            # 3. 插入八叉树
                            insert_frame(root, aug_keypoints, aug_frame_id)

                            # 4. 创建元数据
                            # 注意：这里存储的是旋转后的关键点，以便查询时精确计算距离
                            metadata = FrameMetadata(
                                bvh_file=result["bvh_file"],
                                frame_index=result["frame_index"],
                                frame_id=aug_frame_id,
                                keypoints={
                                    name: np.round(pos, config.JSON_FLOAT_PRECISION)
                                    for name, pos in aug_keypoints.items()
                                },
                            )
                            metadata_list.append(metadata)

                        total_frames += 1
                        file_frame_count += 1

                        if verbose and total_frames % 100 == 0:
                            print(f"  已处理 {total_frames} 帧 (x{len(rot_configs)} 增强)...", end='\r')
                    else:
                        error_count += 1
                        if verbose:
                            print(f"  警告: 无法加载帧 {result['frame_index']}: {result['error']}")
                
                file_elapsed = time.time() - file_start_time
                total_elapsed = time.time() - total_start_time
                
                if verbose:
                    avg = file_elapsed / file_frame_count if file_frame_count else 0
                    print(f"  ✓ 完成 {Path(bvh_file).name}: {file_frame_count} 帧")
                    print(f"  文件用时: {file_elapsed:.2f} 秒 (平均: {avg:.4f} 秒/帧)")
                
                # 清理当前文件的 BVH 缓存，释放内存
                clear_specific_file(bvh_file)
                
                # 强制垃圾回收
                gc.collect()
                
                # 可选：显示内存使用情况
                if verbose:
                    try:
                        import psutil
                        process = psutil.Process()
                        mem_mb = process.memory_info().rss / 1024 / 1024
                        print(f"  [内存] 当前占用: {mem_mb:.1f} MB")
                    except ImportError:
                        pass
        
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
    
    # 4. 统计树结构信息
    if verbose:
        print("\n步骤4: 统计树结构...")
        node_count = count_nodes(root)
        leaf_count = count_leaf_nodes(root)
        max_frames_in_leaf = get_max_frames_in_leaf(root)
        
        print(f"  总节点数: {node_count}")
        print(f"  叶节点数: {leaf_count}")
        print(f"  单个叶节点最大帧数: {max_frames_in_leaf}")
    
    # 5. 保存模型
    if verbose:
        print("\n步骤5: 保存模型...")
    
    save_tree(root, model_tree_path, show_progress=verbose)
    save_metadata(metadata_list, model_metadata_path)
    
    if verbose:
        # 检查大小（如果需要则附加 .npz）
        real_tree_path = model_tree_path
        if not str(model_tree_path).endswith('.npz'):
             real_tree_path += '.npz'
             
        if Path(real_tree_path).exists():
            tree_size = Path(real_tree_path).stat().st_size / 1024 / 1024
            print(f"  八叉树: {real_tree_path} ({tree_size:.2f} MB)")
        else:
            print(f"  八叉树: {model_tree_path} (保存成功)")
            
        metadata_size = Path(model_metadata_path).stat().st_size / 1024 / 1024
        print(f"  元数据: {model_metadata_path} ({metadata_size:.2f} MB)")
    
    if verbose:
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)


def train_model(data_dir: str = "data_train/", 
                model_tree_path: str = "model.npz",
                model_metadata_path: str = "model.pkl",
                verbose: bool = True,
                num_workers: Optional[int] = None) -> None:
    
    if not config.ENABLE_MULTI_TREE:
        train_single_tree(
            data_dir=data_dir,
            model_tree_path=model_tree_path,
            model_metadata_path=model_metadata_path,
            rotation_config=None,
            verbose=verbose,
            num_workers=num_workers
        )
        return
    
    if verbose:
        print("=" * 80)
        print(f"多树训练模式 - {len(config.ROTATION_CONFIGS)}棵树")
        print("=" * 80)
    
    rotation_configs = create_custom_rotation_configs(config.ROTATION_CONFIGS)
    
    base_tree_path = Path(model_tree_path).stem
    # 处理 .npz 扩展名（如果存在）
    if base_tree_path.endswith('.npz'):
        base_tree_path = base_tree_path[:-4]
        
    base_metadata_path = Path(model_metadata_path).stem
    output_dir = Path(model_tree_path).parent
    
    total_start_time = time.time()
    
    for rot_config in rotation_configs:
        # 确保保留 .npz 扩展名
        tree_filename = rot_config.get_model_filename(base_tree_path)
        if not tree_filename.endswith('.npz'):
            tree_filename += '.npz'
            
        tree_path = str(output_dir / tree_filename)
        metadata_path = str(output_dir / rot_config.get_metadata_filename(base_metadata_path))
        
        train_single_tree(
            data_dir=data_dir,
            model_tree_path=tree_path,
            model_metadata_path=metadata_path,
            rotation_config=rot_config,
            verbose=verbose,
            num_workers=num_workers
        )
        
        if verbose:
            print()
    
    total_elapsed = time.time() - total_start_time
    
    if verbose:
        print("=" * 80)
        print("所有树训练完成！")
        print(f"总用时: {total_elapsed:.2f} 秒")


def count_nodes(node) -> int:
    """递归统计节点数"""
    count = 1
    for child in node.children.values():
        count += count_nodes(child)
    return count


def count_leaf_nodes(node) -> int:
    """递归统计叶节点数"""
    if not node.children:
        return 1
    count = 0
    for child in node.children.values():
        count += count_leaf_nodes(child)
    return count


def get_max_frames_in_leaf(node) -> int:
    """获取单个叶节点中的最大帧数"""
    if not node.children:
        return len(node.frame_ids)
    max_frames = 0
    for child in node.children.values():
        max_frames = max(max_frames, get_max_frames_in_leaf(child))
    return max_frames


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="训练帧检索模型")
    parser.add_argument("--data-dir", default="data_train/", help="数据目录路径")
    parser.add_argument("--output-tree", default="model.npz", help="输出树文件路径")
    parser.add_argument("--output-metadata", default="model.pkl", help="输出元数据文件路径")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--workers", type=int, default=None, help="并行加载进程数")
    
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
