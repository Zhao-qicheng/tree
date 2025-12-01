"""
多树训练脚本示例：演示如何建立多个旋转坐标系的八叉树。

这是一个示例脚本，展示如何扩展现有的train.py来支持多树索引。
实际集成时，需要将此逻辑整合到train.py中。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from rotation_utils import create_default_rotation_configs, RotationConfig
from data_loader import load_all_bvh_files, get_bvh_frame_count, load_keypoints_from_bvh
from data_structures import FrameMetadata
from octree_builder import create_root_node, insert_frame, save_tree, save_metadata
from octree_node import ActionTreeNode
import config


def train_multi_tree_model(
    data_dir: str = "data_train/",
    output_dir: str = "models/",
    rotation_configs: List[RotationConfig] = None,
    verbose: bool = True
) -> Dict[int, tuple[str, str]]:
    """
    训练多个旋转坐标系的八叉树模型。
    
    参数:
        data_dir: 训练数据目录
        output_dir: 模型输出目录
        rotation_configs: 旋转配置列表（None则使用默认配置）
        verbose: 是否打印详细信息
    
    返回:
        字典 {tree_id: (tree_path, metadata_path)}
    """
    if rotation_configs is None:
        rotation_configs = create_default_rotation_configs()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    num_trees = len(rotation_configs)
    
    if verbose:
        print("=" * 80)
        print(f"多树训练模式 - {num_trees}棵树")
        print("=" * 80)
        for cfg in rotation_configs:
            print(f"  {cfg}")
        print()
    
    # 1. 扫描BVH文件
    if verbose:
        print(f"步骤1: 扫描 {data_dir} 目录...")
    
    bvh_files = load_all_bvh_files(data_dir)
    if verbose:
        print(f"找到 {len(bvh_files)} 个BVH文件\n")
    
    # 2. 为每个旋转配置训练一棵树
    model_paths = {}
    
    for cfg in rotation_configs:
        if verbose:
            print("=" * 80)
            print(f"训练树 {cfg.tree_id}: {cfg.axis}轴 {cfg.angle}°")
            print("=" * 80)
        
        tree_start_time = time.time()
        
        # 创建根节点
        root = create_root_node()
        metadata_list: List[FrameMetadata] = []
        total_frames = 0
        
        # 加载并插入所有帧
        for bvh_file in bvh_files:
            frame_count = get_bvh_frame_count(bvh_file)
            
            if verbose:
                print(f"\n处理文件: {Path(bvh_file).name} ({frame_count} 帧)")
            
            for frame_index in range(frame_count):
                # 加载原始关键点
                keypoints = load_keypoints_from_bvh(frame_index, bvh_file)
                
                # 应用旋转变换
                rotated_keypoints = cfg.rotate(keypoints)
                
                # 生成帧ID（所有树使用相同的ID）
                from train import generate_frame_id
                frame_id = generate_frame_id(bvh_file, frame_index)
                
                # 插入到树中
                insert_frame(root, rotated_keypoints, frame_id)
                
                # 保存元数据（使用旋转后的坐标）
                metadata = FrameMetadata(
                    bvh_file=bvh_file,
                    frame_index=frame_index,
                    frame_id=frame_id,
                    keypoints={
                        name: np.round(pos, config.JSON_FLOAT_PRECISION)
                        for name, pos in rotated_keypoints.items()
                    }
                )
                metadata_list.append(metadata)
                
                total_frames += 1
                
                if verbose and total_frames % 100 == 0:
                    print(f"  已处理 {total_frames} 帧...", end='\r')
        
        if verbose:
            print(f"\n完成! 总帧数: {total_frames}")
        
        # 保存模型
        tree_path = str(output_path / cfg.get_model_filename())
        metadata_path = str(output_path / cfg.get_metadata_filename())
        
        if verbose:
            print(f"\n保存模型...")
        
        save_tree(root, tree_path, show_progress=verbose)
        save_metadata(metadata_list, metadata_path)
        
        tree_elapsed = time.time() - tree_start_time
        
        if verbose:
            tree_size = Path(tree_path).stat().st_size / 1024 / 1024
            metadata_size = Path(metadata_path).stat().st_size / 1024 / 1024
            print(f"\n  ✓ 树 {cfg.tree_id} 训练完成")
            print(f"  八叉树: {tree_path} ({tree_size:.2f} MB)")
            print(f"  元数据: {metadata_path} ({metadata_size:.2f} MB)")
            print(f"  用时: {tree_elapsed:.2f} 秒\n")
        
        model_paths[cfg.tree_id] = (tree_path, metadata_path)
    
    if verbose:
        print("=" * 80)
        print("所有树训练完成！")
        print("=" * 80)
        total_size = sum(
            Path(p[0]).stat().st_size + Path(p[1]).stat().st_size
            for p in model_paths.values()
        ) / 1024 / 1024
        print(f"总模型大小: {total_size:.2f} MB")
        print(f"平均每棵树: {total_size/num_trees:.2f} MB")
    
    return model_paths


def demo_multi_tree_training():
    """演示多树训练功能。"""
    print("\n" + "=" * 80)
    print("多树训练演示")
    print("=" * 80)
    
    # 使用小规模配置进行演示
    demo_configs = create_default_rotation_configs()[:3]  # 只用3棵树
    
    print("\n注意: 这只是一个演示，使用3棵树进行快速测试")
    print("实际应用建议使用5棵树以获得更好的效果\n")
    
    # 检查数据目录
    if not Path("data_train").exists():
        print("错误: data_train/ 目录不存在")
        print("请确保训练数据已准备好")
        return
    
    try:
        model_paths = train_multi_tree_model(
            data_dir="data_train/",
            output_dir="models_multi/",
            rotation_configs=demo_configs,
            verbose=True
        )
        
        print("\n✅ 多树训练演示完成！")
        print("\n生成的模型文件:")
        for tree_id, (tree_path, meta_path) in model_paths.items():
            print(f"  树{tree_id}:")
            print(f"    - {tree_path}")
            print(f"    - {meta_path}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行演示
    demo_multi_tree_training()
