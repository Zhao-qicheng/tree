import os
import json
import pickle
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
from pathlib import Path

import config
from flat_octree import FlatOctree
from data_structures import FrameMetadata
from utils.visualize_npy import H36M_CONNECTIONS, get_pose_objects

# =============================================================================
# 1. 数据加载与预处理
# =============================================================================

DEFAULT_TREE = "model.npz"
DEFAULT_METADATA = "model.pkl"
DEFAULT_LABELS = "leaf_labels.json"

def load_data():
    """集中加载所有必要的数据"""
    print(f"正在加载模型: {DEFAULT_TREE}")
    tree = FlatOctree.load(DEFAULT_TREE)
    
    print(f"正在加载元数据: {DEFAULT_METADATA}")
    with open(DEFAULT_METADATA, 'rb') as f:
        metadata_list = pickle.load(f)
    metadata_map = {m.frame_id: m for m in metadata_list}
    
    # 获取叶节点并统计
    leaf_indices = []
    for i in range(tree.num_nodes):
        if tree.children_start[i] == tree.children_start[i+1]:
            leaf_indices.append(i)
            
    # 加载标签
    labels = {}
    if os.path.exists(DEFAULT_LABELS):
        with open(DEFAULT_LABELS, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            
    # 转换为 DataFrame 方便表格显示
    table_data = []
    for idx in leaf_indices:
        f_start = tree.frame_ids_start[idx]
        f_end = tree.frame_ids_start[idx + 1]
        table_data.append({
            "idx": idx,
            "count": f_end - f_start,
            "label": labels.get(str(idx), "---")
        })
    
    df = pd.DataFrame(table_data).sort_values(by="count", ascending=False)
    return tree, metadata_map, df

# 全局变量
TREE, METADATA_MAP, LEAF_DF = load_data()

# =============================================================================
# 2. Dash 应用布局
# =============================================================================

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='current-frame-idx', data=0), # 存储当前叶节点内的帧索引
    html.H1("八叉树叶节点动作探索器", style={'textAlign': 'center', 'fontFamily': '微软雅黑', 'color': '#2c3e50', 'padding': '20px'}),
    
    html.Div([
        # 左侧面板：骨骼可视化
        html.Div([
            html.Div([
                html.Span("当前选择: ", style={'fontWeight': 'bold'}),
                html.Span(id='selected-leaf-info', children="请在右侧选择叶节点", style={'color': '#e67e22'})
            ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '10px'}),
            
            # 骨骼可视化区域（带左右按钮）
            html.Div([
                # 左侧按钮
                html.Button('←', id='prev-btn', n_clicks=0, style={
                    'width': '50px', 'height': '100px', 'fontSize': '24px', 'cursor': 'pointer',
                    'backgroundColor': '#ffffff', 'border': '1px solid #bdc3c7', 'borderRadius': '4px'
                }),
                
                # 中间图表
                html.Div([
                    dcc.Loading(
                        id="loading-pose",
                        type="circle",
                        children=dcc.Graph(id='leaf-pose-graph', style={'height': '70vh'})
                    ),
                    html.Div(id='frame-counter', style={'textAlign': 'center', 'fontWeight': 'bold', 'marginTop': '5px'})
                ], style={'flex': '1'}),

                # 右侧按钮
                html.Button('→', id='next-btn', n_clicks=0, style={
                    'width': '50px', 'height': '100px', 'fontSize': '24px', 'cursor': 'pointer',
                    'backgroundColor': '#ffffff', 'border': '1px solid #bdc3c7', 'borderRadius': '4px'
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 20px'}),
        
        # 右侧面板：列表与控制
        html.Div([
             html.Div([
                html.Label("动作名称设置:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Input(id='action-name-input', type='text', placeholder="输入动作名称...", style={'width': '60%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #bdc3c7'}),
                html.Button('保存名称', id='save-label-btn', n_clicks=0, style={
                    'marginLeft': '10px', 'padding': '8px 20px', 'backgroundColor': '#27ae60', 'color': 'white', 
                    'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'
                }),
                html.Div(id='save-status', style={'marginTop': '5px', 'fontSize': '0.9em'})
            ], style={'padding': '15px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #dcdde1', 'borderRadius': '8px', 'marginBottom': '20px'}),

            dash_table.DataTable(
                id='leaf-table',
                columns=[
                    {"name": "编号", "id": "idx"},
                    {"name": "帧数", "id": "count"},
                    {"name": "动作名称", "id": "label"}
                ],
                data=LEAF_DF.to_dict('records'),
                page_size=15,
                sort_action="native",
                filter_action="native",
                row_selectable='single',
                selected_rows=[0],
                style_table={'height': '60vh', 'overflowY': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '8px', 'fontFamily': '微软雅黑'},
                style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'}
            )
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'padding': '0 10px'})
], style={'backgroundColor': '#f4f7f6', 'minHeight': '100vh'})

# =============================================================================
# 3. 回调函数
# =============================================================================

@app.callback(
    Output('current-frame-idx', 'data'),
    [Input('leaf-table', 'selected_rows'),
     Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks')],
    [State('current-frame-idx', 'data'),
     State('leaf-table', 'data')]
)
def handle_navigation(selected_rows, prev_clicks, next_clicks, current_idx, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # 如果切换了叶节点，重置索引为 0
    if trigger_id == 'leaf-table':
        return 0
    
    if not selected_rows:
        return 0
        
    leaf_idx = table_data[selected_rows[0]]['idx']
    max_frames = len(TREE.get_frame_ids(leaf_idx))
    
    if trigger_id == 'prev-btn':
        return max(0, current_idx - 1)
    elif trigger_id == 'next-btn':
        return min(max_frames - 1, current_idx + 1)
    
    return current_idx

@app.callback(
    [Output('leaf-pose-graph', 'figure'),
     Output('selected-leaf-info', 'children'),
     Output('action-name-input', 'value'),
     Output('frame-counter', 'children')],
    [Input('current-frame-idx', 'data'),
     Input('leaf-table', 'selected_rows')],
    [State('leaf-table', 'data')]
)
def update_pose_display(current_idx, selected_rows, table_data):
    # 初始加载或无选择时，默认选中第一行
    if not selected_rows:
        selected_rows = [0]
    if not table_data:
        return go.Figure(), "数据加载中...", "", "0 / 0"
    
    row_data = table_data[selected_rows[0]]
    leaf_idx = row_data['idx']
    label = row_data['label']
    
    frame_ids = TREE.get_frame_ids(leaf_idx)
    num_frames = len(frame_ids)
    
    if num_frames == 0:
        return go.Figure(), f"Leaf #{leaf_idx} (无有效数据)", "", "0 / 0"

    # 仅加载当前帧
    target_fid = frame_ids[current_idx]
    if target_fid not in METADATA_MAP:
        return go.Figure(), f"Leaf #{leaf_idx} (数据缺失: {target_fid})", "", f"{current_idx+1} / {num_frames}"
        
    m = METADATA_MAP[target_fid]
    
    # 从原始 NPY 文件加载数据，保证 17 点完整性
    try:
        source_path = m.source_file
        if not os.path.exists(source_path):
             pose_raw = np.array([m.keypoints[name] for name in config.KEYPOINT_NAMES])
        else:
            full_data = np.load(source_path)
            pose_raw = full_data[m.frame_index]
            if pose_raw.shape != (17, 3):
                pose_raw = pose_raw.reshape(17, 3)
    except:
        pose_raw = np.array([m.keypoints[name] for name in config.KEYPOINT_NAMES])

    # 骨架对齐：中心化 + 旋转对齐（与 action_space_explorer.py 保持一致）
    def align_skeleton(frame):
        """骨架对齐：中心化 + 旋转使骨盆平行于X轴"""
        hip = frame[0]
        centered = frame - hip
        # 计算骨盆向量：从左胯(4)指向右胯(1)
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
    
    # 确定八叉树决策点的索引
    active_joints = set()
    for pair in config.JOINT_PAIRS:
        active_joints.update(pair)
    active_indices = [i for i, name in enumerate(config.KEYPOINT_NAMES) if name in active_joints]
    other_indices = [i for i, name in enumerate(config.KEYPOINT_NAMES) if name not in active_joints]
    
    # 1. 绘制普通关节点 (黑色 + 编号)
    scatter_other = go.Scatter3d(
        x=pose[other_indices, 0], y=pose[other_indices, 1], z=pose[other_indices, 2],
        mode='markers+text',
        marker=dict(size=2, color='black'),
        text=[str(i) for i in other_indices],
        textposition="top center",
        name='普通点'
    )
    
    # 2. 绘制八叉树决策点 (橙色 + 编号)
    scatter_active = go.Scatter3d(
        x=pose[active_indices, 0], y=pose[active_indices, 1], z=pose[active_indices, 2],
        mode='markers+text',
        marker=dict(size=2, color='#e67e22'),
        text=[str(i) for i in active_indices],
        textposition="top center",
        name='决策点'
    )
    
    # 3. 绘制骨骼连线 (红色粗线，与 action_space_explorer.py 一致)
    bones_traces = []
    for start_j, end_j in H36M_CONNECTIONS:
        x = [pose[start_j, 0], pose[end_j, 0], None]
        y = [pose[start_j, 1], pose[end_j, 1], None]
        z = [pose[start_j, 2], pose[end_j, 2], None]
        bones_traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color='red', width=4),
            showlegend=False
        ))
    
    # 坐标范围设为 1000（与 action_space_explorer.py 一致）
    ax_range = 1000
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-ax_range, ax_range], visible=False),
            yaxis=dict(range=[-ax_range, ax_range], visible=False),
            zaxis=dict(range=[-ax_range, ax_range], visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        title=f"运动员: {Path(m.source_file).parts[-3]} | 文件: {Path(m.source_file).name} | 帧: {m.frame_index}",
        showlegend=True,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02)
    )
    
    fig = go.Figure(data=[scatter_other, scatter_active] + bones_traces, layout=layout)
    
    info_text = f"Leaf #{leaf_idx} | 总帧数: {num_frames} | 状态: {'已命名' if label != '---' else '待分类'}"
    input_val = label if label != "---" else ""
    counter_text = f"{current_idx + 1} / {num_frames}"
    
    return fig, info_text, input_val, counter_text

@app.callback(
    [Output('leaf-table', 'data'),
     Output('save-status', 'children')],
    [Input('save-label-btn', 'n_clicks')],
    [State('action-name-input', 'value'),
     State('leaf-table', 'selected_rows'),
     State('leaf-table', 'data')]
)
def save_label(n_clicks, new_name, selected_rows, current_data):
    if n_clicks == 0 or not selected_rows:
        return dash.no_update, ""
    
    row_idx = selected_rows[0]
    leaf_idx = current_data[row_idx]['idx']
    
    # 加载 JSON 并保存
    labels = {}
    if os.path.exists(DEFAULT_LABELS):
        with open(DEFAULT_LABELS, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            
    labels[str(leaf_idx)] = new_name
    
    with open(DEFAULT_LABELS, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
        
    # 更新内存中的数据
    current_data[row_idx]['label'] = new_name
    
    return current_data, f"✔️ 成功保存 #{leaf_idx} 为 '{new_name}'"

if __name__ == '__main__':
    print("八叉树探索器启动中...")
    print("请访问: http://127.0.0.1:8050")
    app.run(debug=True, use_reloader=False)
