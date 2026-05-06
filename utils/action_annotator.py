import os
import sys
import json
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go

# 导入项目中通用的基础算法，用以进行测试帧与历史帧一样的环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npy_loader import normalize_skeleton

# 定义画图连线规范
H36M_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
    (12, 13), (14, 15), (15, 16)
]

# ======= 动作字典常量 =======
JUMP_TYPES = [
    {"label": "Axel (A)", "value": "A"},
    {"label": "Toeloop (T)", "value": "T"},
    {"label": "Loop (Lo)", "value": "Lo"},
    {"label": "Salchow (S)", "value": "S"},
    {"label": "Flip (F)", "value": "F"},
    {"label": "Lutz (Lz)", "value": "Lz"}
]

PHASE_STAGES = [
    {"label": "1: 进入与滑行", "value": "1"},
    {"label": "2: 起跳前压刃", "value": "2"},
    {"label": "3: 起跳离冰", "value": "3"},
    {"label": "4: 起跳过渡", "value": "4"},
    {"label": "5: 空中旋转", "value": "5"},
    {"label": "6: 打开落冰准备", "value": "6"},
    {"label": "7: 落冰与滑出", "value": "7"}
]

TEMPORAL_FLAGS = [
    {"label": "单帧独立判定 (S)", "value": "S"},
    {"label": "需时序相邻帧 (T)", "value": "T"}
]

ACTION_UNITS = [
    {"label": "H1: 头部左脚旋转", "value": "H1"},
    {"label": "H2: 头部右旋转", "value": "H2"},
    {"label": "H3: 头上仰", "value": "H3"},
    {"label": "H4: 头下低", "value": "H4"},
    {"label": "H5: 头左侧倾斜", "value": "H5"},
    {"label": "H6: 头右侧倾斜", "value": "H6"},
    
    {"label": "T1: 躯干左旋转", "value": "T1"},
    {"label": "T2: 躯干右旋转", "value": "T2"},
    {"label": "T3: 躯干前倾", "value": "T3"},
    {"label": "T4: 躯干后仰", "value": "T4"},
    {"label": "T5: 躯干左倾斜", "value": "T5"},
    {"label": "T6: 躯干右倾斜", "value": "T6"},
    
    {"label": "L1: 左上臂前屈", "value": "L1"},
    {"label": "L2: 左上臂外展", "value": "L2"},
    {"label": "L3: 左上臂后伸", "value": "L3"},
    {"label": "L4: 肘部弯曲", "value": "L4"},
    {"label": "L5: 手掌打开", "value": "L5"},
    {"label": "L6: 手掌握拳", "value": "L6"},
    
    {"label": "R1: 右上臂前屈", "value": "R1"},
    {"label": "R2: 右上臂外展", "value": "R2"},
    {"label": "R3: 右上臂后伸", "value": "R3"},
    {"label": "R4: 肘部弯曲", "value": "R4"},
    {"label": "R5: 手掌打开", "value": "R5"},
    {"label": "R6: 手掌握拳", "value": "R6"},
    
    {"label": "l1: 左髋前屈", "value": "l1"},
    {"label": "l2: 左髋外展", "value": "l2"},
    {"label": "l3: 左髋后伸", "value": "l3"},
    {"label": "l4: 左髋内收", "value": "l4"},
    {"label": "l5: 左膝弯曲", "value": "l5"},
    {"label": "l6: 点冰（脚尖下压）", "value": "l6"},
    {"label": "l7: 勾脚尖", "value": "l7"},
    
    {"label": "r1: 右髋前屈", "value": "r1"},
    {"label": "r2: 右髋外展", "value": "r2"},
    {"label": "r3: 右髋后伸", "value": "r3"},
    {"label": "r4: 右髋内收", "value": "r4"},
    {"label": "r5: 右膝弯曲", "value": "r5"},
    {"label": "r6: 点冰（脚尖下压）", "value": "r6"},
    {"label": "r7: 勾脚尖", "value": "r7"}
]
# ===========================

DB_PATH = 'output/action_templates.json'

def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("读取数据库错误:", e)
    return {}

def save_db(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 借用原来探索系统里一模一样的对齐方法以供待测新样本处理前置
def align_skeleton(frame):
    hip = frame[0]
    centered = frame - hip
    v_hip = centered[4] - centered[1]
    v_hip_xy = np.array([v_hip[0], v_hip[1], 0])
    norm = np.linalg.norm(v_hip_xy)
    if norm < 1e-6:
        aligned = centered
    else:
        v_hip_norm = v_hip_xy / norm
        theta = np.arctan2(v_hip_norm[1], v_hip_norm[0])
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        aligned = centered @ R.T
    
    # 增加归一化处理约束长度比例一致
    aligned = normalize_skeleton(aligned)
    return aligned

# 生成 3D 图形的通用 Helper
def create_pose_figure(aligned_pose, title="动作展示", color='red'):
    if aligned_pose is None or len(aligned_pose) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, scene=dict(aspectmode='cube'))
        return fig
        
    joints_scatter = go.Scatter3d(
        x=aligned_pose[:, 0], y=aligned_pose[:, 1], z=aligned_pose[:, 2],
        mode='markers+text',
        marker=dict(size=4, color='black'),
        text=[str(i) for i in range(17)],
        textposition="top center",
        name='关节点'
    )
    
    bones_lines = []
    for start_j, end_j in H36M_CONNECTIONS:
        x = [aligned_pose[start_j, 0], aligned_pose[end_j, 0], None]
        y = [aligned_pose[start_j, 1], aligned_pose[end_j, 1], None]
        z = [aligned_pose[start_j, 2], aligned_pose[end_j, 2], None]
        bones_lines.append(go.Scatter3d(
            x=x, y=y, z=z, mode='lines', 
            line=dict(color=color, width=5), showlegend=False
        ))
        
    fig = go.Figure(data=[joints_scatter] + bones_lines)
    ax_range = 1000
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-ax_range, ax_range], visible=False),
            yaxis=dict(range=[-ax_range, ax_range], visible=False),
            zaxis=dict(range=[-ax_range, ax_range], visible=False),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50) 
    )
    return fig

app = dash.Dash(__name__)

# 构建前端布局
app.layout = html.Div([
    html.H1("🌟 动作智能打标机与测试仪", style={'textAlign': 'center', 'fontFamily': '微软雅黑', 'padding': '20px', 'backgroundColor': '#343a40', 'color': 'white', 'margin': 0}),
    
    dcc.Tabs([
        # ====================== [板块 1] : 动作标签大厅 ======================
        dcc.Tab(label='📖 核心动作标注大厅', children=[
            html.Div([
                # 左侧：选区和输入
                html.Div([
                    html.H3("选择动作集群 (Cluster ID)"),
                    dcc.Dropdown(id='template-dropdown', placeholder="加载或提取动作失败，请先在资源探索器执行保存..."),
                    html.Br(),
                    html.Div(id='template-info', style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                    html.Hr(),
                    
                    html.H3("为这个帧进行多维打标："),
                    
                    html.Label("跳跃类型", style={'fontWeight': 'bold', 'marginTop': '10px', 'display': 'block'}),
                    dcc.Dropdown(id='dropdown-jump-type', options=JUMP_TYPES, placeholder="选择跳跃类型..."),
                    
                    html.Label("阶段划分", style={'fontWeight': 'bold', 'marginTop': '10px', 'display': 'block'}),
                    dcc.Dropdown(id='dropdown-stage', options=PHASE_STAGES, placeholder="选择所处阶段..."),
                    
                    html.Label("时序依赖标志", style={'fontWeight': 'bold', 'marginTop': '10px', 'display': 'block'}),
                    dcc.RadioItems(id='radio-temporal-flag', options=TEMPORAL_FLAGS, value="S", inline=True, style={'marginBottom': '10px'}),
                    
                    html.Label("动作单元 (多选)", style={'fontWeight': 'bold', 'marginTop': '10px', 'display': 'block'}),
                    dcc.Dropdown(id='dropdown-action-units', options=ACTION_UNITS, multi=True, placeholder="可选择多项组合..."),
                    
                    html.Div(id='label-preview', style={'marginTop': '15px', 'padding': '10px', 'fontFamily': 'monospace', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    
                    html.Button("💾 录入并更名", id='save-label-btn', n_clicks=0, style={'marginTop': '15px', 'padding': '10px 20px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '5px', 'fontWeight': 'bold'}),
                    html.Div(id='save-result-msg', style={'marginTop': '15px', 'color': 'blue'})
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxSizing': 'border-box'}),
                
                # 右侧：看这具躯壳长什么样
                html.Div([
                    dcc.Graph(id='template-pose-graph', style={'height': '75vh'})
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ]),

        # ====================== [板块 2] : 新品动作比令人判决 ======================
        dcc.Tab(label='🔬 单帧实时诊断与配型', children=[
            html.Div([
                html.H3("指派本地 .npy 数据源路径进行定级", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Input(id='test-file-path', type='text', placeholder="请输入绝对或相对路径（例：./data/npy/Skater/Axel/1.npy）", style={'width': '50%', 'padding': '10px'}),
                    html.Button("📥 载入并检索帧数", id='load-test-btn', n_clicks=0, style={'marginLeft': '10px', 'padding': '10px'})
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("选取测试帧号进行瞬时比对: ", style={'fontWeight': 'bold'}),
                    html.Div(id='slider-container', children=[
                         dcc.Slider(id='frame-slider', min=0, max=100, step=1, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'width': '80%', 'margin': 'auto'}),
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f1f3f5', 'borderRadius': '10px', 'width': '80%', 'margin': 'auto', 'marginBottom': '20px'}),

                # 显示判定结果超大字号
                html.Div(id='inference-result-text', style={'textAlign': 'center', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#c92a2a', 'marginBottom': '20px', 'height': '40px'}),

                # 双屏展览
                html.Div([
                    html.Div([
                        html.H3("👉 你的测试输入 (提纯对齐后)", style={'textAlign': 'center', 'color': 'gray'}),
                        dcc.Graph(id='test-input-graph', style={'height': '55vh'})
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H3("👈 匹配上的字典范本", style={'textAlign': 'center', 'color': '#2b8a3e'}),
                        dcc.Graph(id='matched-template-graph', style={'height': '55vh'})
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ])
            ], style={'padding': '30px'})
        ])
    ])
])

# ======= 回调大厅：读取选项与展示 =======
@app.callback(
    [Output('template-dropdown', 'options'),
     Output('template-dropdown', 'value')],
    Input('template-dropdown', 'id') # 这个假的触发器让它每次页面载入时查一次库
)
def initialize_dropdown(_):
    db = load_db()
    if not db:
        return [], None
    
    options = []
    # 按照ID排序展示
    for cid in sorted(db.keys(), key=lambda x: int(x)):
        info = db[cid]
        options.append({'label': f"簇 {cid} ({info['label']})", 'value': cid})
        
    return options, list(db.keys())[0]

# 实时预览回调
@app.callback(
    Output('label-preview', 'children'),
    [Input('dropdown-jump-type', 'value'),
     Input('dropdown-stage', 'value'),
     Input('radio-temporal-flag', 'value'),
     Input('dropdown-action-units', 'value')]
)
def update_preview(jump, stage, temporal, units):
    j_str = jump if jump else "?"
    s_str = stage if stage else "?"
    t_str = temporal if temporal else "S"
    u_str = "[" + ",".join(units) + "]" if units else "[]"
    
    preview_text = f"{j_str}-{s_str}-{t_str}-{u_str}"
    return f"🏷️ 预览编码格式: {preview_text}"

@app.callback(
    [Output('template-pose-graph', 'figure'),
     Output('template-info', 'children'),
     Output('dropdown-jump-type', 'value'),
     Output('dropdown-stage', 'value'),
     Output('radio-temporal-flag', 'value'),
     Output('dropdown-action-units', 'value')],
    Input('template-dropdown', 'value'),
    prevent_initial_call=False
)
def render_template(cluster_id):
    if cluster_id is None:
         return dash.no_update, "无展示数据。", None, None, "S", []
    
    db = load_db()
    if cluster_id not in db:
        return dash.no_update, "分类字典已损坏或缺失", None, None, "S", []
        
    data = db[cluster_id]
    skeleton = np.array(data['skeleton'])
    
    fig = create_pose_figure(skeleton, title=f"选定模型 - ID {cluster_id}", color='deepskyblue')
    
    info_text = f"📍 模板标识: 【 {data.get('label', '未标注')} 】\n"
    info_text += f"📂 萃取来源: {data.get('source_file', '未知')}\n"
    info_text += f"🎞️ 所在原帧: 第 {data.get('frame_idx', '未知')} 帧\n"
    
    metadata = data.get('metadata', {})
    j_val = metadata.get('jump_type', None)
    s_val = metadata.get('stage', None)
    t_val = metadata.get('temporal_flag', "S")
    u_val = metadata.get('action_units', [])
    
    return fig, info_text, j_val, s_val, t_val, u_val

# 修改保存动作
@app.callback(
    Output('save-result-msg', 'children'),
    Input('save-label-btn', 'n_clicks'),
    [State('template-dropdown', 'value'), 
     State('dropdown-jump-type', 'value'),
     State('dropdown-stage', 'value'),
     State('radio-temporal-flag', 'value'),
     State('dropdown-action-units', 'value')],
    prevent_initial_call=True
)
def save_custom_label(n_clicks, cid, jump, stage, temporal, units):
    if not cid:
        return "目标字典未选定！"
    if not jump or not stage:
        return "⚠️ 跳跃类型和阶段划分为必填项！"
        
    j_str = jump
    s_str = stage
    t_str = temporal if temporal else "S"
    u_str = "[" + ",".join(units) + "]" if units else "[]"
    
    new_label = f"{j_str}-{s_str}-{t_str}-{u_str}"
    
    db = load_db()
    if cid in db:
        db[cid]['label'] = new_label
        db[cid]['metadata'] = {
            'jump_type': jump,
            'stage': stage,
            'temporal_flag': temporal,
            'action_units': units if units else []
        }
        save_db(db)
        return f"已成功为字典 {cid} 指派新编码：【{new_label}】! 页面刷新后左侧列表变动。"
    return "字典记录定位失败！"

# ======= 测试与单帧匹配回调 =======
# 1. 载入滑块上限设定
@app.callback(
    [Output('slider-container', 'children')],
    Input('load-test-btn', 'n_clicks'),
    State('test-file-path', 'value'),
    prevent_initial_call=True
)
def load_test_file(n_clicks, file_path):
    if not file_path or not os.path.exists(file_path):
        return [html.Div("无效的数据路径！请检查", style={'color': 'red'})]
    
    try:
        data = np.load(file_path)
        frame_cnt = data.shape[0]
        new_slider = dcc.Slider(
            id='frame-slider', min=0, max=frame_cnt-1, step=1, value=0, 
            marks={0: '0', frame_cnt-1: str(frame_cnt-1)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
        return [new_slider]
    except Exception as e:
        return [html.Div(f"解析 .npy 遇到错误：{e}", style={'color': 'red'})]


# 2. 划动拉条或首次加载，直接执行超猛的“距离排查算法”以确认同伴
@app.callback(
    [Output('test-input-graph', 'figure'),
     Output('matched-template-graph', 'figure'),
     Output('inference-result-text', 'children')],
    Input('frame-slider', 'value'),
    State('test-file-path', 'value'),
    prevent_initial_call=True
)
def run_model_inference(frame_idx, file_path):
    if frame_idx is None or not os.path.exists(file_path):
        return dash.no_update, dash.no_update, "尚未装入有效的源数据"
        
    # a. 读取那悲惨并尚未鉴定身份的新帧
    try:
        data = np.load(file_path)
        raw_pose = data[frame_idx]
    except Exception as e:
         return dash.no_update, dash.no_update, f"抽取该阵列错误: {e}"

    # 它也必须经过社会大熔炉统一被我们洗干净
    test_aligned_pose = align_skeleton(raw_pose)
    
    # b. 拿他与 500 名选秀池逐一比较 (最小欧氏距离 MSE)
    db = load_db()
    if not db:
        return dash.no_update, dash.no_update, "知识库不存在！请先前往 3D 浏览器主系统保存提取一波结果。"
        
    best_match_cid = None
    best_dist = float('inf')
    best_template_skeleton = None
    best_label = ""
    
    # 将拉平后的数据进行快速全矩阵迭代
    test_flat = test_aligned_pose.flatten()
    
    for cid, info in db.items():
        tmpl_skeleton = np.array(info['skeleton'])
        tmpl_flat = tmpl_skeleton.flatten()
        # 欧氏距离
        dist = np.linalg.norm(test_flat - tmpl_flat)
        if dist < best_dist:
            best_dist = dist
            best_match_cid = cid
            best_template_skeleton = tmpl_skeleton
            best_label = info['label']
            
    # c. 匹配上了，生成左右双画板
    fig_left = create_pose_figure(test_aligned_pose, title=f"送检动作: {os.path.basename(file_path)} (第{frame_idx}帧)", color='gray')
    fig_right = create_pose_figure(best_template_skeleton, title=f"词典标样: 类别 {best_match_cid} 标准帧", color='#2b8a3e')
    
    verdict = f"判定完成！！ 匹配类型 ➔【 {best_label} 】 ⚡ 距离分差: {best_dist:.2f}"
    
    return fig_left, fig_right, verdict


if __name__ == '__main__':
    print("双子座系统 - [标注查阅仪] 正在全速启动！")
    print("请访问网址: http://127.0.0.1:8051 (独立于探究器)")
    # 我们用 8051 端口避免与之前 8050 那个程序冲撞（即使它在后台没关掉）
    app.run(debug=True, port=8051)
