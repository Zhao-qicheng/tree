"""
测试脚本：演示八叉树动作识别系统的完整功能流程。

本脚本展示了从数据加载、模型训练到预测评估的完整流程。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loader import load_keypoints_from_bvh, load_multiple_frames
from data_structures import BodyKeypoints, coerce_body_keypoints
from octree_builder import (
    create_root_node,
    insert_sample,
    finalize_tree,
    save_tree,
    load_tree,
)
from inference import predict_action
from evaluation import evaluate_accuracy, confusion_matrix
import config
from query import query


def test_data_loading():
    """测试1: 数据加载功能"""
    print("=" * 80)
    print("测试1: 数据加载功能")
    print("=" * 80)
    
    # 从BVH文件加载单个帧的关键点
    print("\n1.1 加载单个帧的关键点:")
    keypoints = load_keypoints_from_bvh(0, bvh_file='data/walk.bvh')
    print(f"   加载的关键点数量: {len(keypoints)}")
    for joint_name, pos in keypoints.items():
        print(f"   {joint_name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # 加载多个帧
    print("\n1.2 加载多个帧的关键点:")
    frames_data = load_multiple_frames([0, 5, 10], bvh_file='data/walk.bvh')
    print(f"   加载的帧数量: {len(frames_data)}")
    
    return keypoints, frames_data


def test_data_structures():
    """测试2: 数据结构转换"""
    print("\n" + "=" * 80)
    print("测试2: 数据结构转换")
    print("=" * 80)
    
    # 从字典创建 BodyKeypoints
    keypoints_dict = load_keypoints_from_bvh(0, bvh_file='data/walk.bvh')
    body_keypoints = coerce_body_keypoints(keypoints_dict)
    
    print("\n2.1 BodyKeypoints 对象:")
    print(f"   Hips: {body_keypoints.Hips}")
    print(f"   LeftHand: {body_keypoints.LeftHand}")
    print(f"   RightHand: {body_keypoints.RightHand}")
    
    # 转换为字典
    print("\n2.2 转换回字典:")
    keypoints_dict_2 = body_keypoints.as_dict()
    print(f"   字典键: {list(keypoints_dict_2.keys())}")
    
    return body_keypoints


def test_tree_building():
    """测试3: 八叉树构建"""
    print("\n" + "=" * 80)
    print("测试3: 八叉树构建")
    print("=" * 80)
    
    # 创建根节点
    print("\n3.1 创建根节点:")
    root = create_root_node()
    print(f"   根节点深度: {root.depth}")
    print(f"   根节点关键点数量: {len(root.bboxes)}")
    print(f"   根节点总样本数: {root.total_samples}")
    
    # 加载训练数据
    print("\n3.2 加载训练样本:")
    bvh_file = 'data/walk.bvh'
    train_frames = [0, 5, 10, 15, 20]  # 选择几个帧作为训练样本
    
    training_samples = []
    for frame_idx in train_frames:
        keypoints = load_keypoints_from_bvh(frame_idx, bvh_file=bvh_file)
        training_samples.append((keypoints, "walk"))  # 标签为 "walk"
    
    print(f"   加载了 {len(training_samples)} 个训练样本")
    
    # 插入样本到树中
    print("\n3.3 插入样本到八叉树:")
    samples_metadata = []
    
    for idx, (keypoints, label) in enumerate(training_samples):
        leaf_node, combination_indices = insert_sample(root, keypoints, label)
        samples_metadata.append({
            "sample_name": f"walk_frame_{train_frames[idx]}",
            "label": label,
            "frame_index": train_frames[idx],
            "combination_indices": combination_indices,
        })
        print(f"   样本 {idx+1}: 标签={label}, 叶节点深度={leaf_node.depth}, "
              f"路径长度={len(combination_indices)}")
    
    # 最终化树结构
    print("\n3.4 最终化树结构（解析标签）:")
    finalize_tree(root)
    print(f"   根节点标签: {root.resolved_label}")
    print(f"   根节点总样本数: {root.total_samples}")
    
    # 统计树结构信息
    def count_nodes(node):
        count = 1
        for _, child in node.iter_children():
            count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(root)
    print(f"   树中总节点数: {total_nodes}")
    
    return root, samples_metadata


def test_model_persistence(root):
    """测试4: 模型持久化"""
    print("\n" + "=" * 80)
    print("测试4: 模型持久化")
    print("=" * 80)
    
    # 保存模型
    model_path = "test_tree.json"
    print(f"\n4.1 保存模型到: {model_path}")
    save_tree(root, model_path)
    
    # 检查文件是否存在
    if Path(model_path).exists():
        file_size = Path(model_path).stat().st_size
        print(f"   模型文件已保存，大小: {file_size} 字节")
    
    # 加载模型
    print(f"\n4.2 从文件加载模型:")
    loaded_root = load_tree(model_path)
    print(f"   加载的根节点深度: {loaded_root.depth}")
    print(f"   加载的根节点标签: {loaded_root.resolved_label}")
    print(f"   加载的根节点总样本数: {loaded_root.total_samples}")
    
    return loaded_root, model_path

def test_evaluation(root):
    """测试6: 模型评估"""
    print("\n" + "=" * 80)
    print("测试6: 模型评估")
    print("=" * 80)
    
    # 创建测试样本（使用不同的帧）
    print("\n6.1 准备测试样本:")
    test_samples = []
    test_frames = [2, 7, 12, 17, 22]
    
    for frame_idx in test_frames:
        try:
            keypoints = load_keypoints_from_bvh(frame_idx, bvh_file='data/walk.bvh')
            test_samples.append((keypoints, "walk"))  # 假设这些都是 walk 动作
        except Exception as e:
            print(f"   跳过帧 {frame_idx}: {e}")
    
    print(f"   准备了 {len(test_samples)} 个测试样本")
    
    # 评估准确率
    print("\n6.2 评估准确率:")
    accuracy_result = evaluate_accuracy(root, test_samples)
    print(f"   总样本数: {accuracy_result['total']}")
    print(f"   正确数: {accuracy_result['correct']}")
    print(f"   准确率: {accuracy_result['accuracy']:.2%}")
    
    # 混淆矩阵
    print("\n6.3 混淆矩阵:")
    cm = confusion_matrix(root, test_samples)
    for true_label, pred_counts in cm.items():
        print(f"   真实标签 '{true_label}':")
        for pred_label, count in pred_counts.items():
            print(f"     -> 预测为 '{pred_label}': {count} 次")


def test_multiple_actions(root):
    """测试7: 多动作类别（如果数据允许）"""
    print("\n" + "=" * 80)
    print("测试7: 多动作类别训练")
    print("=" * 80)
    
    # 尝试加载另一个BVH文件作为不同的动作
    print("\n7.1 尝试加载另一个动作:")
    other_bvh = 'data/01_01.bvh'
    
    if Path(other_bvh).exists():
        try:
            # 加载几个帧作为新动作的样本
            new_action_frames = [0, 5, 10]
            for frame_idx in new_action_frames:
                keypoints = load_keypoints_from_bvh(frame_idx, bvh_file=other_bvh)
                leaf_node, _ = insert_sample(root, keypoints, "action_01")
                print(f"   插入动作 'action_01' 的帧 {frame_idx}")
            
            # 重新最终化树
            finalize_tree(root)
            print(f"\n   重新解析后的根节点标签: {root.resolved_label}")
            print(f"   根节点总样本数: {root.total_samples}")
            
        except Exception as e:
            print(f"   无法加载另一个动作: {e}")
    else:
        print(f"   文件 {other_bvh} 不存在，跳过此测试")


def save_training_metadata(samples_metadata, output_path="test_train_samples.json"):
    """保存训练样本元数据（供 query.py 使用）"""
    metadata = {
        "samples": samples_metadata,
        "total_samples": len(samples_metadata),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练样本元数据已保存到: {output_path}")


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("八叉树动作识别系统 - 完整功能测试")
    print("=" * 80)
    
    try:
        # 测试1: 数据加载
        keypoints, frames_data = test_data_loading()
        
        # 测试2: 数据结构
        body_keypoints = test_data_structures()
        
        # 测试3: 树构建
        root, samples_metadata = test_tree_building()
        
        # 测试4: 模型持久化
        loaded_root, model_path = test_model_persistence(root)

        # 测试6: 评估
        test_evaluation(loaded_root)
        
        # 测试7: 多动作类别
        test_multiple_actions(root)
        
        # 保存训练元数据
        save_training_metadata(samples_metadata)
        
        # 保存最终模型
        final_model_path = "test_tree_final.json"
        save_tree(root, final_model_path)
        print(f"\n最终模型已保存到: {final_model_path}")
        
        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)
        print("\n生成的文件:")
        print(f"  - {model_path} (中间模型)")
        print(f"  - {final_model_path} (最终模型)")
        print(f"  - test_train_samples.json (训练样本元数据)")
        print("\n你可以使用以下命令进行查询:")
        print(f"  python query.py")
        print("\n或者直接使用:")
        print(f"  from query import query")
        print(f"  query(query_frame=7, model_path='{final_model_path}')")
        print("\n测试查询")
        query(query_frame=7, model_path='test_tree_final.json')
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

