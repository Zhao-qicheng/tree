import os
import sys
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# 添加父目录到路径以导入 config 和 npy_loader
sys.path.append(str(Path(__file__).parent.parent))
import config
from npy_loader import load_keypoints_from_npy, normalize_skeleton, align_orientation

def load_first_frame(npy_path):
    # 直接加载原始数据，不做任何预处理
    data = np.load(npy_path)
    return data[10] # 取第一帧

def calculate_bone_lengths(frame):
    lengths = {}
    base_len = getattr(config, 'NORMALIZE_REFERENCE_LENGTH', 100.0)
    
    # 使用 npy_loader 中隐含的拓扑结构（或直接用 config 中的 ratio keys）
    # 这里为了验证，直接用 config 中的 keys
    for key in config.STANDARD_BONE_RATIOS.keys():
        parent_idx, child_idx = map(int, key.split('_'))
        dist = np.linalg.norm(frame[child_idx] - frame[parent_idx])
        lengths[key] = dist
    return lengths

def plot_compare(frame_a_raw, frame_a_retargeted, frame_a_norm, output_file="overlap_verification.html"):
    from plotly.subplots import make_subplots
    
    # 创建骨骼连线
    def get_trace(frame, name, color, opacity=1.0, showlegend=True):
        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
        
        # 关节点
        markers = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color=color),
            name=f"{name} Joints",
            text=[config.KEYPOINT_NAMES[i] for i in range(17)],
            opacity=opacity,
            showlegend=showlegend
        )
        
        # 骨骼连线
        lines_x, lines_y, lines_z = [], [], []
        for key in config.STANDARD_BONE_RATIOS.keys():
            p, c = map(int, key.split('_'))
            lines_x.extend([x[p], x[c], None])
            lines_y.extend([y[p], y[c], None])
            lines_z.extend([z[p], z[c], None])
            
        lines = go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color=color, width=3),
            name=f"{name} Skeleton",
            opacity=opacity,
            showlegend=showlegend
        )
        return [markers, lines]

    # 创建子图: 1行3列
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('1. Raw (Unchanged)', '2. Retargeted Only (No Align)', '3. Final (Align + Retarget)')
    )

    # 准备中心化的原始数据用于对比
    raw_a_centered = frame_a_raw - frame_a_raw[0]

    # 图1: Raw Data
    for trace in get_trace(raw_a_centered, "Raw", "red", 0.8):
        fig.add_trace(trace, row=1, col=1)

    # 图2: Retargeted Only (对比骨骼长度变化，朝向不变)
    # 将 Raw 和 Retargeted 放在一起看
    for trace in get_trace(raw_a_centered, "Raw Shadow", "red", 0.7):
        fig.add_trace(trace, row=1, col=2)
    for trace in get_trace(frame_a_retargeted, "Retargeted", "orange", 0.7):
        fig.add_trace(trace, row=1, col=2)

    # 图3: Final Normalized Data (对齐 + 重定向)
    for trace in get_trace(frame_a_norm, "Final Norm", "orange", 1.0):
        fig.add_trace(trace, row=1, col=3)

    # 统一设置场景
    axis_range = [-900, 900]
    scene_layout = dict(
        xaxis=dict(range=axis_range, title='X'),
        yaxis=dict(range=axis_range, title='Y'),
        zaxis=dict(range=axis_range, title='Z'),
        aspectmode='cube'
    )

    fig.update_layout(
        title="Skeleton Processing Pipeline: Raw -> Retargeted -> Aligned",
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        height=700,
        width=1800,
        legend=dict(x=0.5, y=0.05, orientation="h", xanchor="center")
    )
    
    fig.write_html(output_file)
    print(f"可视化结果已保存至: {output_file}")

def verify_files(file_a, file_b):
    print(f"分析文件: {file_a}")
    
    # 1. 加载原始帧 (第一帧)
    frame_a_raw = load_first_frame(file_a)
    
    # 2. 计算各个阶段的数据
    # 阶段 A: 仅中心化 (用于对比)
    frame_a_centered = frame_a_raw - frame_a_raw[0]
    
    # 阶段 B: 仅重定向 (保留原始朝向)
    # 注意：normalize_skeleton 内部会自动处理中心化
    frame_a_retargeted = normalize_skeleton(frame_a_raw)
    
    # 阶段 C: 旋转对齐 + 重定向 (完整归一化)
    frame_a_aligned = align_orientation(frame_a_raw)
    frame_a_norm = normalize_skeleton(frame_a_aligned)
    
    # 3. 验证骨长 (使用最终结果)
    lens_a = calculate_bone_lengths(frame_a_norm)
    
    print("\n[骨骼长度验证 (Subject A Final)]")
    print(f"{'Bone':<15} | {'Std Len':<10} | {'Actual Len':<10} | {'Diff':<10}")
    print("-" * 55)
    
    for key in config.STANDARD_BONE_RATIOS.keys():
        std_len = config.STANDARD_BONE_RATIOS[key] * config.NORMALIZE_REFERENCE_LENGTH
        la = lens_a[key]
        diff = abs(la - std_len)
        print(f"{key:<15} | {std_len:<10.4f} | {la:<10.4f} | {diff:<10.6f}")
        
    print("-" * 55)

    # 4. 可视化
    plot_compare(frame_a_raw, frame_a_retargeted, frame_a_norm)

if __name__ == "__main__":
    # 直接指定正确的数据根目录
    base_dir = Path("c:/Users/86158/Desktop/八叉树/data/npy")
    file_a = base_dir / "Skater_C" / "Axel" / "Axel_2.npy"
    file_b = base_dir / "Skater_B" / "Axel" / "Axel_1.npy"
    
    if not file_a.exists() or not file_b.exists():
        print("测试文件不存在，请检查路径。")
    else:
        verify_files(str(file_a), str(file_b))
