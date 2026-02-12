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

def calculate_bone_lengths(data_dir, sample_size=70000):
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

def update_config_file(ratios, trunk_len):
    """
    自动更新 config.py 中的比例和参考长度
    """
    config_path = Path(__file__).parent.parent / "config.py"
    if not config_path.exists():
        print(f"警告：未找到 config.py 路径 {config_path}")
        return

    import re
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 更新 NORMALIZE_REFERENCE_LENGTH
    # 匹配 NORMALIZE_REFERENCE_LENGTH: float = XXX.X
    content = re.sub(
        r"(NORMALIZE_REFERENCE_LENGTH:\s*float\s*=\s*)[\d\.]+",
        rf"\g<1>{trunk_len:.1f}",
        content
    )

    # 2. 更新 STANDARD_BONE_RATIOS
    # 找到字典的起始位置
    start_marker = "STANDARD_BONE_RATIOS: Dict[str, float] = {"
    end_marker = "}"
    
    start_idx = content.find(start_marker)
    if start_idx != -1:
        # 寻找对应的结束括号
        # 简单处理：假设字典内没有嵌套括号且以 } 独占行结束或紧跟分号
        end_idx = content.find(end_marker, start_idx)
        
        # 重新生成字典字符串
        new_dict_str = start_marker + "\n"
        for key, ratio in ratios.items():
            parent, child = map(int, key.split('_'))
            name_p = config.KEYPOINT_NAMES[parent]
            name_c = config.KEYPOINT_NAMES[child]
            new_dict_str += f"    '{key}': {ratio:.6f}, # {name_p} -> {name_c}\n"
        
        # 拼接替换
        before = content[:start_idx]
        after = content[end_idx + 1:] # 跳过原本的 }
        content = before + new_dict_str + "}" + after
        
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ 已自动更新 {config_path}")
    print(f"   - NORMALIZE_REFERENCE_LENGTH -> {trunk_len:.1f}")
    print(f"   - STANDARD_BONE_RATIOS 已同步")

def compute_ratios_and_print(all_lengths):
    # 1. 计算中位数长度
    median_lengths = {}
    for child, lengths in all_lengths.items():
        if lengths:
            median_lengths[child] = np.median(lengths)
        else:
            median_lengths[child] = 0.0
            
    # 2. 强制对称性 (取左右平均值)
    for left, right in SYMMETRY_PAIRS:
        l_len = median_lengths.get(left, 0)
        r_len = median_lengths.get(right, 0)
        avg_len = (l_len + r_len) / 2.0
        median_lengths[left] = avg_len
        median_lengths[right] = avg_len
        
    # 3. 计算躯干基准长度
    trunk_len = median_lengths[7] + median_lengths[8] + median_lengths[9]
    print(f"\n[基准] 躯干总长 (Hip->Spine->Chest->Neck): {trunk_len:.4f}")
    
    if trunk_len < 1e-6:
        print("错误：躯干长度过小。")
        return None, 0

    # 4. 生成比例字典
    ratios = {}
    sorted_children = sorted(H36M_TOPOLOGY.keys())
    
    print("\n[计算结果预览]:")
    for child in sorted_children:
        parent = H36M_TOPOLOGY[child]
        key = f"{parent}_{child}"
        ratio = median_lengths[child] / trunk_len
        ratios[key] = ratio
        print(f"    {key}: {ratio:.64f}")
        
    return ratios, trunk_len

def main():
    parser = argparse.ArgumentParser(description="计算数据集的标准骨骼比例")
    parser.add_argument("--data-dir", default=config.FS_JUMP3D_DATA_DIR, help="数据目录")
    parser.add_argument("--sample-size", type=int, default=70000, help="采样帧数")
    parser.add_argument("--auto-update", action="store_true", default=True, help="是否自动更新 config.py")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"数据目录不存在: {args.data_dir}")
        return
        
    all_lengths = calculate_bone_lengths(args.data_dir, args.sample_size)
    ratios, trunk_len = compute_ratios_and_print(all_lengths)
    
    if ratios and args.auto_update:
        update_config_file(ratios, trunk_len)

if __name__ == "__main__":
    main()
