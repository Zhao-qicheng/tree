import os
import json
import dash
from dash import html, dcc, Input, Output, State, ALL
import plotly.graph_objects as go
import numpy as np

import config
from query import _load_model_once

# Human3.6M 骨骼连接关系
SKELETON_CONNECTIONS = [
    ('hip', 'rHip'), ('rHip', 'rKnee'), ('rKnee', 'rAnkle'),
    ('hip', 'lHip'), ('lHip', 'lKnee'), ('lKnee', 'lAnkle'),
    ('hip', 'spine'), ('spine', 'chest'), ('chest', 'neck'), ('neck', 'head'),
    ('chest', 'lShoulder'), ('lShoulder', 'lElbow'), ('lElbow', 'lWrist'),
    ('chest', 'rShoulder'), ('rShoulder', 'rElbow'), ('rElbow', 'rWrist')
]

# 全局变量缓存
app = dash.Dash(__name__, title="兜底归类可视化检验")
tree = None
metadata_list = None
fallback_records = []

def load_data():
    global tree, metadata_list, fallback_records
    if tree is None:
        tree, metadata_list, _ = _load_model_once("models/model.npz", "models/model.pkl", show_progress=False)
    
    if not fallback_records:
        try:
            with open("output/fallback_records.json", 'r', encoding='utf-8') as f:
                fallback_records = json.load(f)
        except FileNotFoundError:
            print("找不到 fallback_records.json。请先运行 evaluate.py")
            fallback_records = []

from utils.visualize_npy import H36M_CONNECTIONS

def create_skeleton_plot(frame_metadata, title="Skeleton", color="blue"):
    from npy_loader import load_keypoints_from_npy
    
    fig = go.Figure()
    
    # 尝试加载包含 17 关节点的完整原始数据以用于完美绘制
    pose_raw = None
    if isinstance(frame_metadata, dict) and "source_file" in frame_metadata:
        # 这是 fallback_records 中的测试帧
        try:
            full_data = np.load(frame_metadata["source_file"])
            pose_raw = full_data[frame_metadata["frame_index"]]
            if pose_raw.shape != (17, 3):
                pose_raw = pose_raw.reshape(17, 3)
        except Exception as e:
            return go.Figure().update_layout(title="Load Error (Test Frame)")
    else:
        # 这是 metadata_list 中的训练帧
        try:
            if os.path.exists(frame_metadata.source_file):
                full_data = np.load(frame_metadata.source_file)
                pose_raw = full_data[frame_metadata.frame_index]
                if pose_raw.shape != (17, 3):
                    pose_raw = pose_raw.reshape(17, 3)
            else:
                pose_raw = np.array([frame_metadata.keypoints[name] for name in config.KEYPOINT_NAMES])
        except Exception:
            return go.Figure().update_layout(title="Format Error (Train Frame)")

    # 1. 骨架对齐 (与 visualize_leaf_nodes 保持一致)
    def align_skeleton(frame):
        hip = frame[0]
        centered = frame - hip
        v_hip = centered[4] - centered[1]
        v_hip_xy = np.array([v_hip[0], v_hip[1], 0])
        norm = np.linalg.norm(v_hip_xy)
        if norm < 1e-6:
            return centered
        v_hip_norm = v_hip_xy / norm
        theta = np.arctan2(v_hip_norm[1], v_hip_norm[0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return centered @ R.T
    
    pose = align_skeleton(pose_raw)
    
    # 进行骨架大小归一化
    from npy_loader import normalize_skeleton
    pose = normalize_skeleton(pose)
    
    # 2. 区分激活(八叉树划分依据)节点和其他节点
    active_joints = set()
    for pair in config.JOINT_PAIRS:
        active_joints.update(pair)
    active_indices = [i for i, name in enumerate(config.KEYPOINT_NAMES) if name in active_joints]
    other_indices = [i for i, name in enumerate(config.KEYPOINT_NAMES) if name not in active_joints]
    
    # 绘制普通关节点 (黑色)
    fig.add_trace(go.Scatter3d(
        x=pose[other_indices, 0], y=pose[other_indices, 1], z=pose[other_indices, 2],
        mode='markers+text',
        marker=dict(size=2, color='black'),
        text=[str(i) for i in other_indices],
        textposition="top center",
        name='普通点'
    ))
    
    # 绘制八叉树关键决策点 (橙色)
    fig.add_trace(go.Scatter3d(
        x=pose[active_indices, 0], y=pose[active_indices, 1], z=pose[active_indices, 2],
        mode='markers+text',
        marker=dict(size=2, color='#e67e22'),
        text=[str(i) for i in active_indices],
        textposition="top center",
        name='决策点'
    ))
    
    # 绘制骨骼连线
    for start_j, end_j in H36M_CONNECTIONS:
        x = [pose[start_j, 0], pose[end_j, 0], None]
        y = [pose[start_j, 1], pose[end_j, 1], None]
        z = [pose[start_j, 2], pose[end_j, 2], None]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color=color, width=4),  # 这里保留 color 以区分左/右画布红蓝色调
            showlegend=False
        ))
    
    ax_range = 1000
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-ax_range, ax_range], visible=False),
            yaxis=dict(range=[-ax_range, ax_range], visible=False),
            zaxis=dict(range=[-ax_range, ax_range], visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False
    )
    return fig


app.layout = html.Div([
    html.H2("八叉树中断兜底算法 (Fallback) 可视化审查", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("选择发生中断的测试帧记录:"),
        dcc.Dropdown(id='record-dropdown', style={'width': '80%', 'marginBottom': '20px'})
    ], style={'textAlign': 'center'}),
    
    html.Div([
        # 左侧面板：被抛弃的原始帧
        html.Div([
            html.H3("原始拦截的测试帧 (Test Frame)", style={'color': 'red', 'textAlign': 'center'}),
            html.Div(id='test-frame-info', style={'whiteSpace': 'pre-line', 'padding': '10px'}),
            dcc.Graph(id='test-frame-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'border': '1px solid #ddd'}),
        
        # 右侧面板：兜底分配的叶节点展示
        html.Div([
            html.H3("强行接收的训练集叶节点 (Fallback Leaf Node)", style={'color': 'blue', 'textAlign': 'center'}),
            html.Div([
                html.Label("切换该叶节点内包含的训练帧:"),
                dcc.Dropdown(id='leaf-frames-dropdown', style={'width': '200px', 'display': 'inline-block', 'marginLeft': '10px'})
            ], style={'padding': '10px'}),
            html.Div(id='train-frame-info', style={'whiteSpace': 'pre-line', 'padding': '0px 10px'}),
            dcc.Graph(id='train-frame-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'border': '1px solid #ddd', 'marginLeft': '2%'})
    ])
])

@app.callback(
    [Output('record-dropdown', 'options'), Output('record-dropdown', 'value')],
    Input('record-dropdown', 'id')
)
def populate_records(_):
    if not fallback_records:
        return [{"label": "没有兜底记录数据", "value": -1}], -1
        
    options = []
    for i, r in enumerate(fallback_records):
        label = f"#{i} | {r['jump_type']} | {r['file_name']} [帧:{r['frame_index']}] -> 中断于第{r['interrupt_depth']}层"
        options.append({"label": label, "value": i})
    
    return options, 0

@app.callback(
    [Output('test-frame-info', 'children'), Output('test-frame-graph', 'figure'),
     Output('leaf-frames-dropdown', 'options'), Output('leaf-frames-dropdown', 'value')],
    Input('record-dropdown', 'value')
)
def update_main_record(record_idx):
    if record_idx is None or record_idx == -1 or not fallback_records:
        return dash.no_update
        
    record = fallback_records[record_idx]
    
    # 构造左侧测试帧信息
    info = f"来源类别: {record['jump_type']}\n"
    info += f"来源文件: {record['file_name']}\n"
    info += f"第 {record['frame_index']} 帧\n\n"
    info += f"发生中断树深: {record['interrupt_depth']}/{config.MAX_DEPTH}\n"
    info += f"欧氏测算距近邻: {record['fallback_distance']:.4f}"
    
    # 绘制左侧图
    fig_test = create_skeleton_plot(record, title=f"Test Frame {record['frame_index']}", color="red")
    
    # 加载该 fallback leaf 的所有包含帧供右侧挑选
    leaf_id = record['fallback_leaf_id']
    fids = tree.get_frame_ids(leaf_id)
    frame_options = [{"label": f"[{i+1}/{len(fids)}] {fid}", "value": fid} for i, fid in enumerate(fids)]
    
    initial_val = frame_options[0]['value'] if frame_options else None
    
    return info, fig_test, frame_options, initial_val

@app.callback(
    [Output('train-frame-info', 'children'), Output('train-frame-graph', 'figure')],
    [Input('leaf-frames-dropdown', 'value'), Input('record-dropdown', 'value')]
)
def update_leaf_frame(train_fid, record_idx):
    if not train_fid or record_idx is None or record_idx == -1:
        return "没有包含帧", go.Figure()
        
    record = fallback_records[record_idx]
    leaf_id = record['fallback_leaf_id']
    
    metadata = next((m for m in metadata_list if m.frame_id == train_fid), None)
    if not metadata:
        return "数据丢失", go.Figure()
        
    info = f"兜底叶节点 ID: {leaf_id}\n"
    info += f"提供特征的训练源文件: {os.path.basename(metadata.source_file)}\n"
    info += f"帧索引: {metadata.frame_index}\n"
    
    fig_train = create_skeleton_plot(metadata, title=f"Train Leaf Frame {metadata.frame_index}", color="blue")
    
    return info, fig_train

if __name__ == '__main__':
    load_data()
    app.run(debug=True, port=8052)
