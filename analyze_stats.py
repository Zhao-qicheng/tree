import os
import numpy as np
from pathlib import Path
import glob

# Human3.6M 骨骼连接关系（父节点 -> 子节点）
H36M_BONE_PAIRS = [
    (0, 7), (7, 8), (8, 9), (9, 10),  # 躯干: Hip -> Spine -> Chest -> Neck -> Head
    (8, 11), (11, 12), (12, 13),      # 左臂: Chest -> LShoulder -> LElbow -> LWrist
    (8, 14), (14, 15), (15, 16),      # 右臂: Chest -> RShoulder -> RElbow -> RWrist
    (0, 4), (4, 5), (5, 6),           # 左腿: Hip -> LHip -> LKnee -> LAnkle
    (0, 1), (1, 2), (2, 3),           # 右腿: Hip -> RHip -> RKnee -> RAnkle
]

def analyze_skeleton_stats(data_dir):
    npy_files = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
    print(f"分析 {len(npy_files)} 个文件...")
    
    all_lengths = {pair: [] for pair in H36M_BONE_PAIRS}
    
    for f in npy_files[:30]:  # 取前30个样本做统计
        data = np.load(f)
        for frame in data[::10]:  # 每10帧取一帧
            for parent_idx, child_idx in H36M_BONE_PAIRS:
                dist = np.linalg.norm(frame[child_idx] - frame[parent_idx])
                all_lengths[(parent_idx, child_idx)].append(dist)
                
    results = {}
    print("\n平均骨骼长度统计 (单位: 原始单位/mm):")
    for pair, lengths in all_lengths.items():
        avg = np.mean(lengths)
        results[pair] = avg
        print(f"骨骼 {pair}: {avg:.2f}")
    
    # 归一化：以脊柱长度 (Hip->Neck) 为基准总长进行归一化，方便在 config 中使用
    # 简化版基准：Hip(0)->Spine(7)->Chest(8)->Neck(9)
    trunk_len = results[(0, 7)] + results[(7, 8)] + results[(8, 9)]
    print(f"\n参考躯干总长 (0->9): {trunk_len:.2f}")
    
    normalized_results = {pair: len_ / trunk_len for pair, len_ in results.items()}
    print("\n归一化比例 (相对于躯干长度 1.0):")
    for pair, val in normalized_results.items():
        print(f"'{pair[0]}_{pair[1]}': {val:.4f},")
        
    return normalized_results

if __name__ == "__main__":
    test_dir = "c:/Users/86158/Desktop/数据集/FS-Jump3D-main/data/npy"
    analyze_skeleton_stats(test_dir)
