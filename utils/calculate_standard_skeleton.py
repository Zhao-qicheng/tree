import os
import sys
import numpy as np
import argparse
import glob
from pathlib import Path

# 添加父目录到路径以导入 config
sys.path.append(str(Path(__file__).parent.parent))
import config

# Human3.6M 骨骼拓扑定义 (Child -> Parent)
# 根节点: Hip (0)
H36M_TOPOLOGY = {
    # 躯干 (Hip -> Spine -> Chest -> Neck -> Head)
    7: 0,
    8: 7,
    9: 8,
    10: 9,
    
    # 左腿 (Hip -> LHip -> LKnee -> LAnkle)
    4: 0,
    5: 4,
    6: 5,
    
    # 右腿 (Hip -> RHip -> RKnee -> RAnkle)
    1: 0, 
    2: 1, 
    3: 2,
    
    # 左臂 (Chest -> LShoulder -> LElbow -> LWrist)
    11: 8,
    12: 11,
    13: 12,
    
    # 右臂 (Chest -> RShoulder -> RElbow -> RWrist)
    14: 8,
    15: 14,
    16: 15
}

# 左右对称对应关系 (Left -> Right)
SYMMETRY_PAIRS = [
    (4, 1),   # LHip -> RHip
    (5, 2),   # LKnee -> RKnee
    (6, 3),   # LAnkle -> RAnkle
    (11, 14), # LShoulder -> RShoulder
    (12, 15), # LElbow -> RElbow
    (13, 16)  # LWrist -> RWrist
]

def load_npy_files(data_dir):
    return glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)

def calculate_bone_lengths(data_dir, sample_size=2000):
    files = load_npy_files(data_dir)
    print(f"找到 {len(files)} 个 NPY 文件")
    
    all_lengths = {child: [] for child in H36M_TOPOLOGY.keys()}
    
    # 随机采样文件
    if len(files) > 0:
        rng = np.random.default_rng(42)
        
        # 简单策略：遍历所有文件，每个文件随机取几帧，直到满足 sample_size
        # 或者随机选文件。这里为了覆盖多样性，随机打乱文件列表，逐个处理直到帧数足够
        rng.shuffle(files)
        
        total_frames_processed = 0
        
        for f in files:
            if total_frames_processed >= sample_size:
                break
                
            try:
                data = np.load(f) # (T, 17, 3)
                n_frames = data.shape[0]
                
                # 每个文件最多取 50 帧，避免某个大文件主导
                frames_to_take =min(50, n_frames)
                indices = rng.choice(n_frames, size=frames_to_take, replace=False)
                
                for idx in indices:
                    frame = data[idx] # (17, 3)
                    
                    # 计算所有骨骼长度
                    for child, parent in H36M_TOPOLOGY.items():
                        dist = np.linalg.norm(frame[child] - frame[parent])
                        all_lengths[child].append(dist)
                        
                total_frames_processed += frames_to_take
                print(f"已处理: {total_frames_processed}/{sample_size} 帧...", end='\r')
                
            except Exception as e:
                print(f"\n无法读取 {f}: {e}")
                
    print(f"\n统计完成，共采样 {total_frames_processed} 帧。")
    return all_lengths

def compute_ratios_and_print(all_lengths):
    # 1. 计算中位数长度
    median_lengths = {}
    for child, lengths in all_lengths.items():
        if lengths:
            median_lengths[child] = np.median(lengths)
        else:
            median_lengths[child] = 0.0
            
    # 2. 强制对称性 (取左右平均值)
    # 先处理对称对
    for left, right in SYMMETRY_PAIRS:
        # 对应的骨骼是 (parent_of_left -> left) 和 (parent_of_right -> right)
        # 例如 LHip(4) 的父是 0, RHip(1) 的父是 0. 长度分别存储在 4 和 1 中
        l_len = median_lengths.get(left, 0)
        r_len = median_lengths.get(right, 0)
        
        avg_len = (l_len + r_len) / 2.0
        median_lengths[left] = avg_len
        median_lengths[right] = avg_len
        
    # 3. 计算躯干基准长度 (Hip -> Spine -> Chest -> Neck)
    # 注意：config.py 中定义的基准是 Hip->Neck 总长度 = 100.0
    # 路径是 0->7->8->9
    trunk_len = median_lengths[7] + median_lengths[8] + median_lengths[9]
    print(f"\n[基准] 躯干总长 (Hip->Spine->Chest->Neck): {trunk_len:.4f} (原始单位)")
    
    if trunk_len < 1e-6:
        print("错误：躯干长度过小，无法归一化。")
        return

    # 4. 生成比例字典
    ratios = {}
    print("\n[结果] 标准骨骼比例 (STANDARD_BONE_RATIOS):")
    print("-" * 40)
    print("STANDARD_BONE_RATIOS: Dict[str, float] = {")
    
    # 按拓扑顺序打印，方便阅读
    sorted_children = sorted(H36M_TOPOLOGY.keys())
    
    for child in sorted_children:
        parent = H36M_TOPOLOGY[child]
        key = f"{parent}_{child}"
        ratio = median_lengths[child] / trunk_len
        ratios[key] = ratio
        
        # 添加注释说明骨骼名称
        name_p = config.KEYPOINT_NAMES[parent]
        name_c = config.KEYPOINT_NAMES[child]
        print(f"    '{key}': {ratio:.6f}, # {name_p} -> {name_c}")
        
    print("}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="计算数据集的标准骨骼比例")
    parser.add_argument("--data-dir", default=config.FS_JUMP3D_DATA_DIR, help="数据目录")
    parser.add_argument("--sample-size", type=int, default=2000, help="采样帧数")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"数据目录不存在: {args.data_dir}")
        return
        
    all_lengths = calculate_bone_lengths(args.data_dir, args.sample_size)
    compute_ratios_and_print(all_lengths)

if __name__ == "__main__":
    main()
