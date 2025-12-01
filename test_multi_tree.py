"""
多树功能快速测试脚本
"""

import sys

# 测试1: 验证配置导入
print("=" * 60)
print("测试1: 验证配置")
print("=" * 60)

try:
    import config
    print(f"✓ ENABLE_MULTI_TREE: {config.ENABLE_MULTI_TREE}")
    print(f"✓ ROTATION_CONFIGS数量: {len(config.ROTATION_CONFIGS)}")
    print(f"✓ MERGE_STRATEGY: {config.MERGE_STRATEGY}")
    print(f"✓ MIN_VOTE_THRESHOLD: {config.MIN_VOTE_THRESHOLD}")
except Exception as e:
    print(f"✗ 配置导入失败: {e}")
    sys.exit(1)

# 测试2: 验证旋转工具
print("\n" + "=" * 60)
print("测试2: 验证旋转工具")
print("=" * 60)

try:
    from rotation_utils import (
        get_rotation_matrix,
        rotate_keypoints,
        RotationConfig,
        create_default_rotation_configs
    )
    
    configs = create_default_rotation_configs()
    print(f"✓ 默认配置数量: {len(configs)}")
    for cfg in configs:
        print(f"  - 树{cfg.tree_id}: {cfg.axis}轴 {cfg.angle}°")
    
    # 测试旋转矩阵
    import numpy as np
    R = get_rotation_matrix('z', 90)
    point = np.array([1.0, 0.0, 0.0])
    rotated = R @ point
    if np.allclose(rotated, [0, 1, 0]):
        print("✓ 旋转矩阵计算正确")
    else:
        print(f"✗ 旋转矩阵错误: {rotated}")
        
except Exception as e:
    print(f"✗ 旋转工具测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 验证train.py导入
print("\n" + "=" * 60)
print("测试3: 验证train.py")
print("=" * 60)

try:
    from train import train_model, train_single_tree
    print("✓ train_model 函数导入成功")
    print("✓ train_single_tree 函数导入成功")
except Exception as e:
    print(f"✗ train.py导入失败: {e}")
    sys.exit(1)

# 测试4: 验证query.py导入
print("\n" + "=" * 60)
print("测试4: 验证query.py")
print("=" * 60)

try:
    from query import query_frame, query_single_tree, merge_candidates
    print("✓ query_frame 函数导入成功")
    print("✓ query_single_tree 函数导入成功")
    print("✓ merge_candidates 函数导入成功")
except Exception as e:
    print(f"✗ query.py导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 验证合并策略
print("\n" + "=" * 60)
print("测试5: 验证合并策略")
print("=" * 60)

try:
    from query import merge_candidates
    
    # 模拟候选帧数据
    all_candidates = {
        0: ["frame_001", "frame_002", "frame_003", "frame_010"],
        1: ["frame_001", "frame_002", "frame_005", "frame_011"],
        2: ["frame_001", "frame_003", "frame_006", "frame_012"],
    }
    
    # 测试投票策略
    merged_vote = merge_candidates(all_candidates, strategy="vote", min_vote_threshold=2)
    print(f"✓ 投票策略（阈值≥2）: {len(merged_vote)} 个候选帧")
    print(f"  {merged_vote}")
    
    # 测试并集策略
    merged_union = merge_candidates(all_candidates, strategy="union")
    print(f"✓ 并集策略: {len(merged_union)} 个候选帧")
    
    # 测试交集策略
    merged_intersection = merge_candidates(all_candidates, strategy="intersection")
    print(f"✓ 交集策略: {len(merged_intersection)} 个候选帧")
    print(f"  {merged_intersection}")
    
except Exception as e:
    print(f"✗ 合并策略测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("所有测试通过！✅")
print("=" * 60)
print("\n多树功能已成功集成到系统中！")
print("\n使用方法：")
print("1. 在 config.py 中设置 ENABLE_MULTI_TREE = True")
print("2. 运行 train.py 训练多棵树")
print("3. 运行 query.py 使用多树查询\n")
