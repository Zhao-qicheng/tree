import os
import numpy as np
import plotly.graph_objects as go
from argparse import ArgumentParser

# Human3.6M 骨骼连接定义（基于标准 H3.6M 格式的索引）
H36M_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
    (12, 13), (14, 15), (15, 16)
]

def get_pose_objects(pose, connections, color='#2E86AB', line_color='#E94F37', show_labels=True):
    """
    生成单个姿态的绘图对象（关节点和骨干连线）。
    
    参数:
        pose: (N, 3) 形状的数组，包含 N 个关节点的 3D 坐标
        connections: 骨骼连接列表
        color: 关节点的颜色
        line_color: 骨干连线的颜色
        show_labels: 是否显示关节点索引
    返回:
        [scatter, lines]: 包含 Plotly 散点对象和线条对象的列表
    """
    # 绘制关节点（散点图）- 增大尺寸、添加边框提升可见度
    scatter = go.Scatter3d(
        x=pose[:, 0], y=pose[:, 1], z=pose[:, 2],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=6,
            color=color,
            line=dict(width=1.5, color='#1a1a1a'),
            opacity=0.95
        ),
        text=[str(i) for i in range(pose.shape[0])] if show_labels else None,
        textposition="top center",
        textfont=dict(size=10, color='#333333', family='Arial'),
        name='关键点'
    )
    
    # 绘制骨干连线 - 加粗线条、使用更醒目的颜色
    x_lines, y_lines, z_lines = [], [], []
    for c1, c2 in connections:
        if c1 < pose.shape[0] and c2 < pose.shape[0]:
            x_lines.extend([pose[c1, 0], pose[c2, 0], None])
            y_lines.extend([pose[c1, 1], pose[c2, 1], None])
            z_lines.extend([pose[c1, 2], pose[c2, 2], None])
            
    lines = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(width=5, color=line_color),
        name='骨架连线'
    )
    return [scatter, lines]

def show_3D_pose_static(pose3d: np.ndarray, connections: list, title: str = "3D 姿态可视化", show_labels: bool = True):
    """
    使用 Plotly 显示单个静态 3D 姿态。
    """
    data_objs = get_pose_objects(pose3d, connections, show_labels=show_labels)
    fig = go.Figure(data=data_objs)

    # 根据当前姿态自动计算坐标轴范围，并添加 15% 的外边距
    x_min, x_max = pose3d[:, 0].min(), pose3d[:, 0].max()
    y_min, y_max = pose3d[:, 1].min(), pose3d[:, 1].max()
    z_min, z_max = pose3d[:, 2].min(), pose3d[:, 2].max()
    pad = max(x_max - x_min, y_max - y_min, z_max - z_min, 200) * 0.15
    x_range = [x_min - pad, x_max + pad]
    y_range = [y_min - pad, y_max + pad]
    z_range = [z_min - pad, z_max + pad]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#2c3e50')),
        width=900, height=900,
        paper_bgcolor='#f8f9fa',
        font=dict(family='Microsoft YaHei, sans-serif', size=12),
        scene=dict(
            xaxis=dict(title='X', range=x_range, visible=False, showgrid=True, gridcolor='#e0e0e0'),
            yaxis=dict(title='Y', range=y_range, visible=False, showgrid=True, gridcolor='#e0e0e0'),
            zaxis=dict(title='Z', range=z_range, visible=False, showgrid=True, gridcolor='#e0e0e0'),
            aspectmode='cube',
            bgcolor='#ffffff',
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.2), center=dict(x=0, y=0, z=0))
        ),
        margin=dict(r=20, l=20, b=20, t=60),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    print(f"正在打开静态可视化窗口: {title}")
    fig.show()

def show_3D_pose_animation(pose3d_seq: np.ndarray, connections: list, title: str = "3D 动画预览", show_labels: bool = False):
    """
    带有自动适配坐标轴的 3D 姿态动画。动画模式默认不显示关节点标签以提升流畅度。
    """
    num_frames = pose3d_seq.shape[0]
    
    # 计算整个序列的全局边界，确保播放过程中视角稳定
    x_min, x_max = np.min(pose3d_seq[:,:,0]), np.max(pose3d_seq[:,:,0])
    y_min, y_max = np.min(pose3d_seq[:,:,1]), np.max(pose3d_seq[:,:,1])
    z_min, z_max = np.min(pose3d_seq[:,:,2]), np.max(pose3d_seq[:,:,2])
    
    # 添加 10% 的平滑外边距
    pad_x = (x_max - x_min) * 0.1 if x_max != x_min else 500
    pad_y = (y_max - y_min) * 0.1 if y_max != y_min else 500
    pad_z = (z_max - z_min) * 0.1 if z_max != z_min else 500

    # 1. 布局配置：包含播放控制、进度条和坐标轴设定
    layout = go.Layout(
        title=dict(text=title, font=dict(size=18, color='#2c3e50')),
        width=900, height=900,
        paper_bgcolor='#f8f9fa',
        font=dict(family='Microsoft YaHei, sans-serif', size=12),
        scene=dict(
            xaxis=dict(title='X', range=[x_min - pad_x, x_max + pad_x], visible=False, showgrid=True, gridcolor='#e0e0e0'),
            yaxis=dict(title='Y', range=[y_min - pad_y, y_max + pad_y], visible=False, showgrid=True, gridcolor='#e0e0e0'),
            zaxis=dict(title='Z', range=[z_min - pad_z, z_max + pad_z], visible=False, showgrid=True, gridcolor='#e0e0e0'),
            aspectmode='cube',
            bgcolor='#ffffff',
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.2), center=dict(x=0, y=0, z=0))
        ),
        margin=dict(r=20, l=20, b=80, t=60),
        # 添加播放/暂停按钮
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶ 播放", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                dict(label="⏸ 暂停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            direction="left", pad={"r": 15, "t": 10}, x=0.1, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor='#ffffff', bordercolor='#dee2e6', borderwidth=1
        )],
        # 添加底部进度条（滑块）
        sliders=[dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=14, color='#495057'), prefix="帧: ", visible=True, xanchor="right"),
            pad=dict(b=15, t=10), len=0.9, x=0.05, y=0,
            steps=[dict(
                args=[[f"frame_{k}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                label=str(k), method="animate"
            ) for k in range(num_frames)],
            bgcolor='#f8f9fa', bordercolor='#dee2e6', borderwidth=1
        )]
    )

    # 2. 生成初始数据（第 0 帧）
    initial_data = get_pose_objects(pose3d_seq[0], connections, show_labels=show_labels)
    
    # 3. 生成每一帧的动画数据
    frames = [go.Frame(data=get_pose_objects(pose3d_seq[k], connections, show_labels=show_labels), name=f"frame_{k}") for k in range(num_frames)]

    # 4. 构建并显示图表
    fig = go.Figure(data=initial_data, layout=layout, frames=frames)
    print("正在浏览器中打开 3D 动画预览...")
    fig.show()

def main():
    # 命令行参数解析
    parser = ArgumentParser(description="用于可视化 .npy 格式 3D 位姿数据的脚本。")
    parser.add_argument("--path", type=str, default=r'D:\work\讨论\FS-Jump3D-main\FS-Jump3D-main\data\npy\Skater_A\Lutz\Lutz_1.npy', help=".npy 文件的路径。")
    parser.add_argument("--frame", type=int, default=-1, help="指定显示的静态帧索引；若设为 -1（默认值）则进入动画模式。")
    parser.add_argument("--center", action="store_true", help="将骨架重心（髋部/节点0）置于原点中心。")
    parser.add_argument("--no-labels", action="store_true", help="隐藏关节点索引标签，使骨架更简洁。")
    
    args = parser.parse_args()

    # 检查文件是否存在，支持相对于当前路径或上级路径搜索
    if not os.path.exists(args.path):
        root_relative = os.path.join('..', args.path)
        if os.path.exists(root_relative): 
            args.path = root_relative
        else:
            print(f"错误: 找不到文件 {args.path}")
            return

    try:
        # 加载数据
        data = np.load(args.path)
        print(f"成功加载数据: {args.path}，形状: {data.shape}")
        
        # 中心化处理逻辑：所有关节点坐标减去根节点（髋部）的坐标
        if args.center:
            print("正在执行中心化：将 Hip (节点 0) 设置为空间原点...")
            data = data - data[:, 0:1, :]

        filename = os.path.basename(args.path)
        title_suffix = " (已中心化)" if args.center else ""

        show_labels = not args.no_labels
        # 根据参数选择显示模式
        if args.frame == -1:
            # 动画模式
            show_3D_pose_animation(data, H36M_CONNECTIONS, title=f"文件: {filename}{title_suffix}")
        else:
            # 单帧静态模式
            if 0 <= args.frame < data.shape[0]:
                show_3D_pose_static(data[args.frame], H36M_CONNECTIONS, title=f"文件: {filename}{title_suffix} | 帧: {args.frame}", show_labels=show_labels)
            else:
                print(f"错误: 帧索引 {args.frame} 超出范围 (该文件总帧数为 {data.shape[0]})。")
            
    except Exception as e:
        print(f"运行时发生错误: {e}")

if __name__ == "__main__":
    main()
