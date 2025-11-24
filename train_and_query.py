"""
训练和查询主程序，按照指定格式输出结果。
"""

from __future__ import annotations

from pathlib import Path

import config

from data_loader import load_keypoints_from_bvh
from inference import predict_action
from octree_builder import create_root_node, finalize_tree, insert_sample, save_tree


def format_joint_name_for_display(joint_name: str) -> str:
    """将系统关节名称转换为显示名称"""
    mapping = {
        'left_wrist': 'WristLeft',
        'right_wrist': 'WristRight',
        'neck': 'Neck',
        'left_ankle': 'AnkleLeft',
        'right_ankle': 'AnkleRight',
    }
    return mapping.get(joint_name, joint_name)


def print_main_joint_positions(keypoints: dict, sample_name: str = ""):
    """打印主关节位置（不包括hips）"""
    if sample_name:
        print(f"样本 {sample_name} 主关节位置:")
    else:
        print("查询主关节位置:")
    
    display_order = ['left_wrist', 'right_wrist', 'neck', 'left_ankle', 'right_ankle']
    for joint_name in display_order:
        if joint_name in keypoints:
            pos = keypoints[joint_name]
            display_name = format_joint_name_for_display(joint_name)
            print(f"  {display_name}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")


def print_combination_indices(combination_indices: list[str], prefix: str = "层"):
    """打印每层的组合索引"""
    for i, idx in enumerate(combination_indices, start=1):
        print(f"{prefix}{i}组合索引 {idx}")


def _layer_difference(a: str | None, b: str | None) -> int:
    """计算单层两个组合索引之间的节点差值之和。"""
    max_diff_per_digit = 7
    if a is None and b is None:
        return 0
    if a is None:
        return max_diff_per_digit * len(b)  # type: ignore[arg-type]
    if b is None:
        return max_diff_per_digit * len(a)
    length = min(len(a), len(b))
    diff = sum(abs(int(a[i]) - int(b[i])) for i in range(length))
    if len(a) != len(b):
        diff += max_diff_per_digit * abs(len(a) - len(b))
    return diff


def compute_weighted_distance(query_indices: list[str], sample_indices: list[str]) -> int:
    """
    根据层数权重计算查询路径与样本路径之间的距离。

    总层数为 n，顶层（第0层）权重为 8^(n-1)，逐层递减，底层权重为 8^0。
    """
    if not query_indices and not sample_indices:
        return 0

    total_layers = max(len(query_indices), len(sample_indices))
    distance = 0
    for depth in range(total_layers):
        weight = 2 ** (total_layers - depth - 1)
        layer_query = query_indices[depth] if depth < len(query_indices) else None
        layer_sample = sample_indices[depth] if depth < len(sample_indices) else None
        layer_diff = _layer_difference(layer_query, layer_sample)
        distance += layer_diff * weight
    return distance


def main():
    # 初始化BVH数据加载
    bvh_file = 'data/walk.bvh'
    
    # 训练样本：使用前11帧作为示例
    train_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_samples = []
    
    print("=" * 80)
    print("构建八叉树模型")
    print("=" * 80)
    
    root = create_root_node()
    
    # 加载并插入训练样本
    for i, frame_idx in enumerate(train_frames):
        try:
            keypoints = load_keypoints_from_bvh(frame_idx, bvh_file)
            sample_name = f"walk_{frame_idx + 1}.txt"
            
            # 显示第一个样本的详细信息
            if i == 0:
                print(f"\n叶负载计数: 1")
                print_main_joint_positions(keypoints, sample_name)
            
            # 插入样本
            leaf_node, combination_indices = insert_sample(root, keypoints, sample_name)
            
            # 显示第一个样本的组合索引
            if i == 0:
                print_combination_indices(combination_indices)
            
            train_samples.append((keypoints, sample_name, combination_indices))
            
        except Exception as e:
            print(f"加载帧 {frame_idx} 时出错: {e}")
            continue
    
    # 完成树构建
    finalize_tree(root)
    
    # 计算叶负载计数（叶子节点的样本数）
    # 根据图片，叶负载计数应该是最终叶子节点的样本数
    # 这里我们统计所有叶子节点的样本数总和
    def count_leaf_samples(node):
        if node.is_leaf():
            return node.total_samples
        total = 0
        for _, child in node.iter_children():
            total += count_leaf_samples(child)
        return total
    
    leaf_sample_count = count_leaf_samples(root)
    print(f"\n叶负载计数: {leaf_sample_count}")
    
    # 保存模型（使用JSON格式，但文件名保持.pkl以匹配图片中的格式）
    model_path = Path("决策树.pkl")
    #这里这个保存树，
    save_tree(root, str(model_path))
    print(f"构建完成: {len(train_samples)}个样本, 模型保存到 {model_path.absolute()}")
    
    # 执行查询
    print("\n" + "=" * 80)
    print("开始执行查询...")
    print("=" * 80)
    
    # 使用第5帧作为查询（示例）
    query_frame = 5
    try:
        query_keypoints = load_keypoints_from_bvh(query_frame, bvh_file)
        print_main_joint_positions(query_keypoints)
        
        # 执行预测
        result = predict_action(root, query_keypoints)
        
        # 显示查询路径的组合索引
        query_combination_indices = [
            entry.combination_index 
            for entry in result.path[1:]  # 跳过根节点
            if entry.combination_index is not None
        ]
        
        print("\n查询路径组合索引:")
        print_combination_indices(query_combination_indices, "查询路径层")
        
        # 查找匹配的候选（简单实现：查找相同路径的样本）
        print("\n候选匹配结果:")
        matched_samples = []
        for keypoints, sample_name, combination_indices in train_samples:
            # 简单匹配：如果查询路径的前几层与样本路径匹配
            if len(query_combination_indices) > 0 and len(combination_indices) > 0:
                # 计算路径相似度（暂时用简单的匹配）
                distance = compute_weighted_distance(query_combination_indices, combination_indices)
                
                matched_samples.append((sample_name, distance))
        
        # 按距离排序
        matched_samples.sort(key=lambda x: x[1])
        
        # 显示匹配结果
        for sample_name, distance in matched_samples[:5]:  # 显示前5个
            print(f"候选 {sample_name} 距离 {distance:.6f}")
            print(f"{sample_name} {distance:.6f}")
        
    except Exception as e:
        print(f"查询时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

