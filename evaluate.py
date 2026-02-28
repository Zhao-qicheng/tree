"""
测试集评估脚本 (evaluate.py)

目的：
使用在训练集上构建的八叉树模型，验证测试集数据是否能成功命中现有的叶子节点。
如果测试集的某帧数据在八叉树下探匹配时，无法在对应层级找到匹配特征（即遇到了未知动作空间），
则认为发生中断（Interrupt），评估将统计这类帧的比例及其中断的平均深度。
"""

import sys
import time
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np

import config
from query import _load_model_once, _normalize_query_input
from data_structures import coerce_body_keypoints, compute_octant, BoundingBox

def check_frame_match(tree, query_keypoints_norm: dict[str, np.ndarray]) -> Tuple[bool, int, int]:
    """
    检查一帧数据在八叉树中的匹配情况。
    
    返回:
        is_hit (bool): 是否成功命中已有叶子节点
        interrupt_depth (int): 如果中断，发生在第几层 (0到MAX_DEPTH-1)；如果成功为 MAX_DEPTH 或叶节点深度
        current_node (int): 最终停留的节点索引
    """
    body = coerce_body_keypoints(query_keypoints_norm)
    keypoint_map = body.as_dict()

    current_node = 0  # 根节点索引
    fallback_used = False
    fallback_depth = -1
    original_interrupt_node = -1
    fallback_distance = 0.0
    
    for depth in range(config.MAX_DEPTH):
        keys, indices = tree.get_children(current_node)
        
        # 如果当前节点没有子节点，说明它已经是训练集建立的天然叶节点，且正常走到了这里 -> 命中!
        if len(keys) == 0:
            return True, depth, current_node, fallback_used, fallback_depth, original_interrupt_node, fallback_distance

        from octree_builder import _get_active_joint_names
        active_names = _get_active_joint_names(depth)
        
        # 动态获取当前节点的包围盒进行空间判断
        octants = tuple(
            compute_octant(keypoint_map[name], BoundingBox.from_tuple(
                (tuple(tree.bboxes[current_node, tree.keypoint_names.index(name), 0:3]),
                 tuple(tree.bboxes[current_node, tree.keypoint_names.index(name), 3:6]))
            ))
            for name in active_names
        )
        
        found_next = False
        for i, key in enumerate(keys):
            if tuple(key[:len(octants)]) == octants:
                current_node = indices[i]
                found_next = True
                break
        
        # 如果中间过程未能找到可匹配的子节点空间（遇到未知数据特征）
        if not found_next:
            # === 最近邻欧氏距离兜底逻辑 (Nearest-Neighbor Fallback) ===
            fallback_used = True
            fallback_depth = depth
            original_interrupt_node = current_node
            
            best_dist = float('inf')
            best_idx = -1
            
            # 使用关节位置
            query_positions = np.array([keypoint_map[name] for name in active_names])  # (N, 3)
            
            # 遍历这棵树真实已有的所有子节点，找出几何中心距离最近的
            for i, child_idx in enumerate(indices):
                # 获取该子节点用于当前深度划分的关轴包围盒中心
                centers = []
                for name in active_names:
                    kp_idx = tree.keypoint_names.index(name)
                    min_b = tree.bboxes[child_idx, kp_idx, 0:3]
                    max_b = tree.bboxes[child_idx, kp_idx, 3:6]
                    centers.append((min_b + max_b) / 2.0)
                
                centers = np.array(centers) # (N, 3)
                
                # 计算与查询点欧式距离的平均值
                dist = np.mean(np.linalg.norm(query_positions - centers, axis=1))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = child_idx
            
            # 强行指派给最近的节点继续往下探
            current_node = best_idx
            fallback_distance = best_dist
            # 此处不 break，让它接管后继续往下循环
    
    # 达到最大深度
    return True, depth, current_node, fallback_used, fallback_depth, original_interrupt_node, fallback_distance


def evaluate_model(model_tree_path: str = "models/model.npz",
                   model_metadata_path: str = "models/model.pkl",
                   test_data_dir: str = None):
    
    if test_data_dir is None:
        test_data_dir = getattr(config, "TEST_DATA_DIR", "data/npy_test")
        
    print("=" * 70)
    print("八叉树模型评估 (泛化能力验证)")
    print(f"测试集路径: {test_data_dir}")
    print(f"树模型文件: {model_tree_path}")
    print("=" * 70)

    if not Path(test_data_dir).exists():
        print(f"\n[错误] 找不到测试集目录 {test_data_dir} ！")
        print("请先执行 `python split_dataset.py` 来生成物理划分好的测试集。")
        sys.exit(1)

    # 1. 加载模型树
    print("正在加载八叉树模型...")
    start_time = time.time()
    try:
        tree, _, _ = _load_model_once(model_tree_path, model_metadata_path, use_cache=True, show_progress=False)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已对训练集完成了 train.py 来建立树模型！")
        sys.exit(1)
        
    print(f"模型加载完毕！包含 {len(tree.node_parent_idx)} 个扁平树节点。耗时 {time.time()-start_time:.2f} 秒\n")

    # 2. 扫描测试集数据
    from npy_loader import load_all_npy_files, get_npy_frame_count, load_keypoints_from_npy, clear_specific_npy_file
    
    try:
        test_files = load_all_npy_files(test_data_dir)
    except FileNotFoundError:
        print(f"测试集目录 {test_data_dir} 为空，中止评估。")
        return
        
    print(f"扫描到 {len(test_files)} 个测试用的 .npy 文件。")
    if not test_files:
        print("测试集为空，中止评估。")
        return

    total_frames = 0
    pure_hit_frames = 0
    miss_frames = 0
    fallback_records = []
    interrupt_depths = []
    
    # 统计信息按跳跃类型拆分
    # {jump_type: {'total': 0, 'hit': 0}}
    stats_by_jump = {}

    print("\n开始执行评估下探 (Traversal)...")
    eval_start_time = time.time()
    
    # 我们避免引入复杂的并行以防止树查找时的潜在竞争，且单帧计算也极快
    for i, file_path in enumerate(test_files, 1):
        num_frames = get_npy_frame_count(file_path)
        
        # 提取类别 / 跳跃名称作为显示
        from npy_loader import get_npy_metadata
        meta = get_npy_metadata(file_path)
        j_type = meta.get('jump_type', 'Unknown')
        if j_type not in stats_by_jump:
            stats_by_jump[j_type] = {'total': 0, 'hit': 0}
            
        file_hits = 0
        for frame_idx in range(num_frames):
            try:
                # 加载并提取数据
                raw_kps = load_keypoints_from_npy(frame_idx, file_path, align=False, normalize=False)
                # 执行与查询同等级别的归一化
                norm_kps = _normalize_query_input(raw_kps)
                
                # 下探查询
                success, final_depth, final_node, is_fallback, fallback_depth, original_node, f_dist = check_frame_match(tree, norm_kps)
                
                total_frames += 1
                stats_by_jump[j_type]['total'] += 1
                
                # 用户要求：保留原来的模型命中率，即没有任何兜底的纯天然命中
                if not is_fallback:
                    pure_hit_frames += 1
                    file_hits += 1
                    stats_by_jump[j_type]['hit'] += 1
                else:
                    miss_frames += 1
                    interrupt_depths.append(fallback_depth)
                    
                    # 记录被兜底的数据以供 Web UI 查看
                    record = {
                        "skater": meta.get('skater', 'Unknown'),
                        "jump_type": j_type,
                        "file_name": Path(file_path).name,
                        "frame_index": int(frame_idx),
                        "interrupt_depth": int(fallback_depth),
                        "interrupt_node": int(original_node),
                        "fallback_leaf_id": int(final_node),
                        "fallback_distance": float(f_dist),
                        "source_file": file_path
                    }
                    fallback_records.append(record)
                    
            except Exception as e:
                # 忽略加载错误或无用的空帧
                pass
                
        # 内存释放
        clear_specific_npy_file(file_path)
        gc.collect()

        print(f"  [{i}/{len(test_files)}] 文件 {Path(file_path).name} : {num_frames}帧, 初始自然命中 {file_hits} 帧.", end='\r')

    eval_time = time.time() - eval_start_time
    print(f"\n\n遍历完成！总用时: {eval_time:.2f} 秒 (约 {total_frames/eval_time:.1f} 帧/秒)")
    
    # 导出兜底异常日志
    import json
    fallback_log_path = "output/fallback_records.json"
    os.makedirs("output", exist_ok=True)
    with open(fallback_log_path, 'w', encoding='utf-8') as f:
        json.dump(fallback_records, f, ensure_ascii=False, indent=2)

    # 3. 输出统计报告
    print("\n" + "=" * 70)
    print(" 评估统计报告 ")
    print("=" * 70)
    print(f"测试总帧数:        {total_frames}")
    if total_frames == 0:
        return
        
    hit_rate = pure_hit_frames / total_frames * 100
    print(f"成功命中已有叶节点(不含兜底): {pure_hit_frames} 帧")
    print(f"发生路径中断(启用算法兜底):   {miss_frames} 帧")
    print(f"\n--> 【初始模型识别命中率】:   {hit_rate:.2f} %")
    
    print(f"\n[兜底分配报告]")
    print(f"  针对中断的 {miss_frames} 帧已启动欧氏距离最近邻算法全部兜底分配。")
    print(f"  兜底详细日志已输出至: {fallback_log_path}")
    print(f"  请使用 `python visualize_fallback_web.py` 开启可视化审查这些干预归类。")
    
    if interrupt_depths:
        avg_depth = sum(interrupt_depths) / len(interrupt_depths)
        print(f"\n[中断深度剖析]")
        print(f"  平均抛弃深度: 第 {avg_depth:.2f} 层 (总深度 {config.MAX_DEPTH} 层)")
        print(f"  抛弃点深度分布:")
        depth_counts = Counter(interrupt_depths)
        for d in sorted(depth_counts.keys()):
            ratio = depth_counts[d] / miss_frames * 100
            print(f"    第 {d} 层兜底干预: {depth_counts[d]} 帧 ({ratio:.1f}%)")
        print("    *(注: 深度如果在前几层中断，说明基础动作差异极大；而在非常深处才中断，体现了受末端关节摆放的影响)*")
        
    if stats_by_jump:
        print("\n[细分类别初始命中率]")
        for jt, s_dict in stats_by_jump.items():
            t_f = s_dict['total']
            if t_f > 0:
                h_f = s_dict['hit']
                print(f"  - 跳跃类别 {jt:<8} | 初始命中率: {h_f/t_f*100:5.1f}% ({h_f}/{t_f})")
                
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估八叉树模型的泛化与覆盖性能")
    parser.add_argument("--test-dir", type=str, default=None, help="测试数据集路径 (默认读取 config.TEST_DATA_DIR)")
    parser.add_argument("--model-tree", default="models/model.npz", help="树模型文件路径")
    parser.add_argument("--model-metadata", default="models/model.pkl", help="元数据文件路径")
    
    args = parser.parse_args()
    
    try:
        evaluate_model(
            model_tree_path=args.model_tree,
            model_metadata_path=args.model_metadata,
            test_data_dir=args.test_dir
        )
    except KeyboardInterrupt:
        print("\n评估被用户中断。")
    except Exception as e:
        print(f"\n评估出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
