"""
帧检索系统测试脚本。

测试完整的训练和查询流程，验证系统功能。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List
import numpy as np

from data_loader import load_all_bvh_files, get_bvh_frame_count, load_keypoints_from_bvh
from train import train_model, generate_frame_id
from query import query_frame, query_from_bvh
from similarity import compute_weighted_distance
import config


def test_training():
    """测试1: 训练模型"""
    print("\n" + "=" * 80)
    print("测试1: 训练模型")
    print("=" * 80)
    
    # 训练模型
    start_time = time.time()
    
    train_model(
        data_dir="data/",
        model_tree_path="model.tree",
        model_metadata_path="model.pkl",
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\n训练耗时: {training_time:.2f} 秒")
    
    # 检查文件是否生成
    assert Path("model.tree").exists(), "模型树文件未生成"
    assert Path("model.pkl").exists(), "元数据文件未生成"
    
    print("\n✓ 训练测试通过")


def test_exact_match_query():
    """测试2: 精确匹配查询"""
    print("\n" + "=" * 80)
    print("测试2: 精确匹配查询")
    print("=" * 80)
    
    # 选择一个训练集中的帧进行查询
    bvh_files = load_all_bvh_files("data/")
    test_file = bvh_files[0]  # 使用第一个文件
    test_frame_index = 10  # 使用第10帧
    
    print(f"\n查询训练集中的帧: {Path(test_file).name}, 帧索引: {test_frame_index}")
    
    # 执行查询
    start_time = time.time()
    results = query_from_bvh(
        bvh_file=test_file,
        frame_index=test_frame_index,
        model_tree_path="model.tree",
        model_metadata_path="model.pkl",
        top_k=5,
        verbose=True
    )
    query_time = time.time() - start_time
    
    print(f"\n查询耗时: {query_time:.4f} 秒")
    
    # 验证结果
    assert len(results) > 0, "未返回任何结果"
    
    # 第一个结果应该是精确匹配
    best_match = results[0]
    expected_frame_id = generate_frame_id(test_file, test_frame_index)
    
    print(f"\n验证精确匹配:")
    print(f"  期望帧ID: {expected_frame_id}")
    print(f"  返回帧ID: {best_match.frame_metadata.frame_id}")
    print(f"  距离值: {best_match.distance:.6f}")
    print(f"  是否精确匹配: {best_match.is_exact_match}")
    
    # 断言第一个结果是精确匹配
    assert best_match.frame_metadata.frame_id == expected_frame_id, \
        f"最佳匹配帧ID不正确: {best_match.frame_metadata.frame_id} != {expected_frame_id}"
    
    assert best_match.is_exact_match, \
        f"应该是精确匹配，但距离为: {best_match.distance}"
    
    print("\n✓ 精确匹配测试通过")


def test_fuzzy_query():
    """测试3: 模糊查询（查询不在训练集中的帧）"""
    print("\n" + "=" * 80)
    print("测试3: 模糊查询")
    print("=" * 80)
    
    # 选择一个训练集中的帧，然后添加噪声
    bvh_files = load_all_bvh_files("data/")
    test_file = bvh_files[0]
    test_frame_index = 20
    
    print(f"\n基于帧 {Path(test_file).name}, 帧索引: {test_frame_index}")
    
    # 加载原始帧
    original_keypoints = load_keypoints_from_bvh(test_frame_index, test_file)
    
    # 添加小的随机噪声（使其不完全匹配）
    noisy_keypoints = {}
    noise_scale = 0.5  # 噪声幅度（单位：原始单位）
    
    for joint_name, pos in original_keypoints.items():
        noise = np.random.randn(3) * noise_scale
        noisy_keypoints[joint_name] = pos + noise
    
    print(f"添加噪声（标准差: {noise_scale}）后进行查询...")
    
    # 执行查询
    start_time = time.time()
    results = query_frame(
        query_keypoints=noisy_keypoints,
        model_tree_path="model.tree",
        model_metadata_path="model.pkl",
        top_k=5,
        verbose=True
    )
    query_time = time.time() - start_time
    
    print(f"\n查询耗时: {query_time:.4f} 秒")
    
    # 验证结果
    assert len(results) > 0, "未返回任何结果"
    
    # 最相似的帧应该是原始帧或其相邻帧
    best_match = results[0]
    expected_frame_id = generate_frame_id(test_file, test_frame_index)
    
    print(f"\n模糊查询结果:")
    print(f"  原始帧ID: {expected_frame_id}")
    print(f"  最相似帧ID: {best_match.frame_metadata.frame_id}")
    print(f"  距离值: {best_match.distance:.6f}")
    print(f"  相似度: {best_match.similarity_score:.4f}")
    
    # 由于添加了噪声，不应该是精确匹配
    assert not best_match.is_exact_match, "添加噪声后不应该是精确匹配"
    
    # 但最相似的帧应该来自同一个文件
    assert Path(best_match.frame_metadata.bvh_file).name == Path(test_file).name, \
        "最相似的帧应该来自同一个文件"
    
    print("\n✓ 模糊查询测试通过")


def test_cross_file_query():
    """测试4: 跨文件查询（验证不同文件的帧检索）"""
    print("\n" + "=" * 80)
    print("测试4: 跨文件查询")
    print("=" * 80)
    
    bvh_files = load_all_bvh_files("data/")
    
    if len(bvh_files) < 2:
        print("\n跳过: 数据目录中少于2个BVH文件")
        return
    
    # 使用第一个文件的某一帧查询
    query_file = bvh_files[0]
    query_frame_index = 5
    
    print(f"\n使用文件 {Path(query_file).name} 的帧 {query_frame_index} 进行查询")
    
    results = query_from_bvh(
        bvh_file=query_file,
        frame_index=query_frame_index,
        model_tree_path="model.tree",
        model_metadata_path="model.pkl",
        top_k=10,
        verbose=False
    )
    
    # 统计结果中来自不同文件的帧数
    file_distribution = {}
    for result in results:
        filename = Path(result.frame_metadata.bvh_file).name
        file_distribution[filename] = file_distribution.get(filename, 0) + 1
    
    print(f"\nTop-10结果的文件分布:")
    for filename, count in sorted(file_distribution.items(), key=lambda x: -x[1]):
        print(f"  {filename}: {count} 帧")
    
    print("\n✓ 跨文件查询测试通过")


def test_performance():
    """测试5: 性能统计"""
    print("\n" + "=" * 80)
    print("测试5: 性能统计")
    print("=" * 80)
    
    from octree_builder import load_tree, load_metadata
    
    # 加载模型
    tree = load_tree("model.tree")
    metadata_list = load_metadata("model.pkl")
    
    print(f"\n模型统计:")
    print(f"  训练集帧数: {len(metadata_list)}")
    
    # 统计树信息
    from train import count_nodes, count_leaf_nodes, get_max_frames_in_leaf
    
    node_count = count_nodes(tree)
    leaf_count = count_leaf_nodes(tree)
    max_frames = get_max_frames_in_leaf(tree)
    
    print(f"  八叉树总节点数: {node_count}")
    print(f"  八叉树叶节点数: {leaf_count}")
    print(f"  单叶节点最大帧数: {max_frames}")
    
    # 多次查询测试平均速度
    bvh_files = load_all_bvh_files("data/")
    test_file = bvh_files[0]
    
    n_queries = 10
    query_times = []
    
    print(f"\n执行 {n_queries} 次查询测试...")
    for i in range(n_queries):
        frame_index = i * 5  # 每5帧查询一次
        
        start_time = time.time()
        results = query_from_bvh(
            bvh_file=test_file,
            frame_index=frame_index,
            model_tree_path="model.tree",
            model_metadata_path="model.pkl",
            top_k=5,
            verbose=False
        )
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        print(f"  查询 {i+1}/{n_queries}: {query_time:.4f} 秒", end='\r')
    
    avg_query_time = np.mean(query_times)
    std_query_time = np.std(query_times)
    
    print(f"\n\n查询性能:")
    print(f"  平均查询时间: {avg_query_time:.4f} 秒")
    print(f"  查询时间标准差: {std_query_time:.4f} 秒")
    print(f"  最快查询时间: {np.min(query_times):.4f} 秒")
    print(f"  最慢查询时间: {np.max(query_times):.4f} 秒")
    
    print("\n✓ 性能测试完成")


def main():
    """运行所有测试。"""
    print("\n" + "=" * 80)
    print("帧检索系统完整测试")
    print("=" * 80)
    
    try:
        # 测试1: 训练
        test_training()
        
        # 测试2: 精确匹配
        test_exact_match_query()
        
        # 测试3: 模糊查询
        test_fuzzy_query()
        
        # 测试4: 跨文件查询
        test_cross_file_query()
        
        # 测试5: 性能统计
        test_performance()
        
        print("\n" + "=" * 80)
        print("所有测试通过！✓")
        print("=" * 80)
        
        print("\n生成的文件:")
        print("  - model.tree: 八叉树模型（二进制）")
        print("  - model.pkl: 帧元数据（pickle）")
        
        print("\n使用说明:")
        print("  训练: python train.py --data-dir data/")
        print("  查询: python query.py --bvh-file data/01_01.bvh --frame-index 10")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

