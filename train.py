"""
训练脚本：从所有BVH文件加载数据，构建八叉树索引。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

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


def train_model(data_dir: str = "data/", 
                model_tree_path: str = "model.tree",
                model_metadata_path: str = "model.pkl",
                verbose: bool = True) -> None:
    """
    训练帧检索模型。
    
    参数:
        data_dir: 数据目录路径
        model_tree_path: 八叉树模型保存路径
        model_metadata_path: 元数据保存路径
        verbose: 是否打印详细信息
    """
    if verbose:
        print("=" * 80)
        print("开始训练帧检索模型")
        print("=" * 80)
    
    # 1. 扫描所有BVH文件
    if verbose:
        print(f"\n步骤1: 扫描 {data_dir} 目录...")
    
    bvh_files = load_all_bvh_files(data_dir)
    if verbose:
        print(f"找到 {len(bvh_files)} 个BVH文件:")
        for bvh_file in bvh_files:
            print(f"  - {Path(bvh_file).name}")
    
    # 2. 创建八叉树根节点
    if verbose:
        print("\n步骤2: 创建八叉树根节点...")
    
    root = create_root_node()
    if verbose:
        print(f"根节点创建成功，深度: {root.depth}")
        print(f"八叉树使用的关键点数量: {len(config.OCTREE_KEYPOINT_NAMES)}")
        print(f"关键点: {', '.join(config.OCTREE_KEYPOINT_NAMES)}")
    
    # 3. 遍历所有文件和帧，插入到八叉树
    if verbose:
        print("\n步骤3: 加载并插入所有帧...")
    
    metadata_list: List[FrameMetadata] = []
    total_frames = 0
    error_count = 0
    
    # 记录总开始时间
    total_start_time = time.time()
    
    for bvh_file in bvh_files:
        try:
            frame_count = get_bvh_frame_count(bvh_file)
            if verbose:
                print(f"\n处理文件: {Path(bvh_file).name} ({frame_count} 帧)")
            
            # 记录当前文件开始时间
            file_start_time = time.time()
            file_frame_count = 0
            
            for frame_index in range(frame_count):
                try:
                    # 加载关键点
                    keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
                    
                    # 生成帧ID
                    frame_id = generate_frame_id(bvh_file, frame_index)
                    
                    # 插入到八叉树
                    leaf_node, combination_indices = insert_frame(root, keypoints, frame_id)
                    
                    # 创建元数据
                    # 将坐标精度设置为6位小数
                    rounded_keypoints = {
                        name: np.round(pos, config.JSON_FLOAT_PRECISION)
                        for name, pos in keypoints.items()
                    }
                    
                    metadata = FrameMetadata(
                        bvh_file=bvh_file,
                        frame_index=frame_index,
                        frame_id=frame_id,
                        keypoints=rounded_keypoints
                    )
                    metadata_list.append(metadata)
                    
                    total_frames += 1
                    file_frame_count += 1
                    
                    # 每处理100帧打印一次进度
                    if verbose and total_frames % 100 == 0:
                        print(f"  已处理 {total_frames} 帧...", end='\r')
                
                except Exception as e:
                    error_count += 1
                    if verbose:
                        print(f"  警告: 无法加载帧 {frame_index}: {e}")
            
            # 文件处理完成，输出该文件用时和总时长
            file_elapsed = time.time() - file_start_time
            total_elapsed = time.time() - total_start_time
            
            if verbose:
                print(f"  ✓ 完成 {Path(bvh_file).name}: {file_frame_count} 帧")
                print(f"  文件用时: {file_elapsed:.2f} 秒 (平均: {file_elapsed/file_frame_count:.4f} 秒/帧)")
                print(f"  总时长: {total_elapsed:.2f} 秒")
        
        except Exception as e:
            if verbose:
                print(f"  错误: 无法处理文件 {Path(bvh_file).name}: {e}")
    
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
        
        # 检查是否使用joblib
        try:
            import joblib
            print("  使用joblib序列化（更快、支持压缩）")
        except ImportError:
            print("  使用pickle序列化（建议安装joblib以获得更好性能）")
    
    save_tree(root, model_tree_path)
    if verbose:
        print(f"  八叉树已保存到: {model_tree_path}")
    
    save_metadata(metadata_list, model_metadata_path)
    if verbose:
        print(f"  元数据已保存到: {model_metadata_path}")
        
        # 显示文件大小
        tree_size = Path(model_tree_path).stat().st_size / 1024 / 1024
        metadata_size = Path(model_metadata_path).stat().st_size / 1024 / 1024
        print(f"  树文件大小: {tree_size:.2f} MB")
        print(f"  元数据文件大小: {metadata_size:.2f} MB")
    
    if verbose:
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
    parser.add_argument("--data-dir", default="data/", help="数据目录路径")
    parser.add_argument("--output-tree", default="model.tree", help="输出树文件路径")
    parser.add_argument("--output-metadata", default="model.pkl", help="输出元数据文件路径")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不打印详细信息")
    
    args = parser.parse_args()
    
    try:
        train_model(
            data_dir=args.data_dir,
            model_tree_path=args.output_tree,
            model_metadata_path=args.output_metadata,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
