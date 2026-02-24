import os
import time
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd

# Human3.6M 骨骼连接定义
H36M_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
    (12, 13), (14, 15), (15, 16)
]

# --- 1. 数据处理与特性工程 ---

def align_skeleton(frame):
    """
    骨架对齐函数：
    1. 中心化：将 Hip (节点0) 移至原点。
    2. 旋转对齐：使 HipLeft(1)->HipRight(4) 向量平行于 X 轴，且人体面朝 Y 轴正方向。
    这样做是为了消除不同姿态由于初始朝向不同带来的干扰，让聚类专注于动作形状。
    """
    # 1. 中心化处理
    hip = frame[0]
    centered = frame - hip

    # 2. 旋转对齐
    # 计算骨盆向量：从左胯 (1) 指向右胯 (4)
    v_hip = centered[4] - centered[1]
    # 投影到 XY 平面以忽略垂直方向的倾斜
    v_hip_xy = np.array([v_hip[0], v_hip[1], 0])
    norm = np.linalg.norm(v_hip_xy)
    if norm < 1e-6:
        return centered  # 避免除零

    v_hip_norm = v_hip_xy / norm
    
    # 我们的目标是旋转这个向量使其落在 X 轴 (1, 0, 0) 上
    # 计算当前向量与 X 轴的夹角 theta
    theta = np.arctan2(v_hip_norm[1], v_hip_norm[0])
    
    # 定义旋转矩阵：绕 Z 轴旋转 -theta 度
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # 应用旋转变换
    aligned = centered @ R.T
    return aligned

def load_and_process_data(root_dir='./data/npy/Skater_A/Axel/Axel_1.npy'):
    """
    加载 .npy 数据，执行对齐，并使用 t-SNE 进行降维映射。
    支持两种输入：
    - 单个 .npy 文件路径
    - 包含多个 .npy 文件的目录路径
    """
    all_frames = []
    metadata = []  # 存储每帧的元数据：文件名、帧号、完整路径、运动员ID
    
    print("正在加载并对齐数据...")
    files_processed = 0
    start_time = time.time()
    
    def process_single_file(path):
        """处理单个 .npy 文件"""
        nonlocal files_processed
        data = np.load(path)  # shape: (总帧数, 17个关节, 3个坐标)
        filename = os.path.basename(path)
        
        # 遍历文件中的每一帧进行预处理
        for i in range(data.shape[0]):
            frame = data[i]
            aligned = align_skeleton(frame)
            
            # 路径解析逻辑：根据文件夹结构提取运动员名称
            path_parts = os.path.normpath(path).split(os.sep)
            try:
                npy_index = path_parts.index('npy')
                skater_name = path_parts[npy_index + 1]
            except ValueError:
                skater_name = 'Unknown'
            
            all_frames.append(aligned.flatten())
            metadata.append({
                'file': filename,
                'frame': i,
                'path': path,
                'skater': skater_name
            })
        files_processed += 1
    
    # 判断输入是文件还是目录
    if os.path.isfile(root_dir) and root_dir.endswith('.npy'):
        # 单文件模式
        process_single_file(root_dir)
    elif os.path.isdir(root_dir):
        # 目录模式：递归遍历
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.npy'):
                    path = os.path.join(root, f)
                    process_single_file(path)
    else:
        print(f"错误：路径不存在或不是有效的 .npy 文件/目录: {root_dir}")
        return pd.DataFrame()

    print(f"数据加载完成。处理了 {files_processed} 个文件，共 {len(all_frames)} 帧。耗时: {time.time()-start_time:.2f}s")
    
    X = np.array(all_frames)
    
    # 降维处理第一步：PCA 降噪
    # 作用：将 51 维 (17关节*3) 压缩到 50 维，保留 99% 以上的方差，同时加速后面 t-SNE 的计算。
    print("正在执行 PCA 降噪...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    
    # 降维处理第二步：t-SNE 降维至 3D 空间
    # 作用：捕捉高维特征中的非线性结构（聚类团块），使结果利于 3D 可视化。
    print("正在执行 t-SNE 降维 (这可能需要几分钟重新计算)...")
    # TSNE设定的核心参数：
    # n_components=3: 输出维度
    # perplexity=30: 信息熵，影响团块的紧密程度
    # max_iter=1000: 最大迭代次数
    # init='pca': 初始化方式
    # learning_rate='auto': 学习率
    # n_jobs=-1：开启多线程加速（非常重要，否则会很慢）
    # verbose=1：打印进度，方便查看收敛情况
    # early_exaggeration=20：加大夸张，把类推开
    tsne = TSNE(
        n_components=3, 
        perplexity=50, 
        max_iter=1500, 
        init='pca', 
        learning_rate='auto', 
        early_exaggeration=20,   
        verbose=2)
    
    X_embedded = tsne.fit_transform(X_pca)
    print("t-SNE 降维计算完成。")
    
    # 将降维后的坐标添加到元数据 DataFrame
    df = pd.DataFrame(metadata)
    df['x'] = X_embedded[:, 0]
    df['y'] = X_embedded[:, 1]
    df['z'] = X_embedded[:, 2]
    
    return df

# --- 2. 缓存管理 ---
# 为了避免每次启动都耗费几分钟计算 t-SNE，我们将结果缓存到本地 CSV
CACHE_FILE = 'output/action_space_cache.csv'
if os.path.exists(CACHE_FILE):
    print(f"检测到缓存文件 {CACHE_FILE}，正在快速加载...")
    df = pd.read_csv(CACHE_FILE)
else:
    df = load_and_process_data()
    # 首次计算后自动导出缓存
    df.to_csv(CACHE_FILE, index=False)
    print(f"分析结果已保存至 {CACHE_FILE}")

# --- 3. Dash 应用布局与交互 ---

app = dash.Dash(__name__)

def generate_scatter_figure(dataframe, cluster_labels=None):
    """
    根据数据帧和可选的聚类标签生成左侧 3D 星系图。
    """
    if cluster_labels is None:
        # 默认模式：按索引染色，呈现时间轴上的连续变化
        color_data = dataframe.index
        colorscale = 'Viridis'
        hover_text = "Skater: " + dataframe['skater'] + " | " + dataframe['file'] + " | F:" + dataframe['frame'].astype(str)
    else:
        # 聚类模式：每个类别拥有独立的颜色
        color_data = cluster_labels
        colorscale = 'Jet' # 离散色彩谱，方便区分不同的团块
        hover_text = "Skater: " + dataframe['skater'] + " | " + dataframe['file'] + " | F:" + dataframe['frame'].astype(str) + " | Cluster: " + cluster_labels.astype(str)

    fig = go.Figure(data=[go.Scatter3d(
        x=dataframe['x'], y=dataframe['y'], z=dataframe['z'],
        mode='markers',
        marker=dict(
            size=1, # 极小点，适合展示数万帧数据
            color=color_data,
            colorscale=colorscale,
            opacity=0.6 # 透明度，避免点重叠时遮挡
        ),
        text=hover_text,
        customdata=dataframe.index, # 用于回调中精确定位 DataFrame
        hoverinfo='text'
    )])

    fig.update_layout(
        title=f"3D 动作星系图 (全量数据：{len(dataframe)} 帧)",
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        dragmode='turntable', # 左键旋转，滚轮缩放
        hovermode='closest',
        uirevision='constant_galaxy' # 核心设置：数据更新时（如重新聚类）保持当前视角不重置
    )
    return fig

# 右侧骨骼预览的初始占位
empty_fig = go.Figure()
empty_fig.update_layout(title="请将鼠标置于左图数据点上以预览动作", scene=dict(aspectmode='cube'))

app.layout = html.Div([
    html.H1("花滑动作 3D 空间探索器", style={'textAlign': 'center', 'fontFamily': '微软雅黑'}),
    
    # 交互控制区：设置聚类数量
    html.Div([
        html.Label("聚类数量 (K): ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Input(id='k-input', type='number', value=8, min=2, max=50, step=1, style={'width': '60px', 'marginRight': '10px'}),
        html.Button('执行 K-Means 聚类分析', id='cluster-btn', n_clicks=0, style={'cursor': 'pointer', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'padding': '8px 20px', 'borderRadius': '5px', 'marginRight': '10px'}),
        html.Button('保存当前结果', id='save-btn', n_clicks=0, style={'cursor': 'pointer', 'backgroundColor': '#28A745', 'color': 'white', 'border': 'none', 'padding': '8px 20px', 'borderRadius': '5px', 'marginRight': '10px'}),
        html.Button('分析 K 值趋势 (8-50)', id='analyze-btn', n_clicks=0, style={'cursor': 'pointer', 'backgroundColor': '#17A2B8', 'color': 'white', 'border': 'none', 'padding': '8px 20px', 'borderRadius': '5px'}),
        html.Div(id='cluster-status', style={'marginTop': '10px', 'color': 'gray', 'fontStyle': 'italic'})
    ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #eaeaea'}),

    html.Div([
        # 左侧边栏：动作分布图
        html.Div([
            dcc.Graph(
                id='galaxy-graph',
                figure=generate_scatter_figure(df),
                style={'height': '85vh'},
                clear_on_unhover=True,
                config={'scrollZoom': True, 'displayModeBar': True}
            )
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # 右侧面板：当前帧骨架实时预览与分析图表（互斥显示）
        html.Div([
            # 1. 骨架预览图 (默认显示)
            dcc.Graph(
                id='pose-preview',
                figure=empty_fig,
                style={'height': '85vh', 'display': 'block'} 
            ),
            # 2. K 值分析图表 (默认隐藏)
            dcc.Loading(
                id="loading-metrics",
                type="default",
                children=dcc.Graph(id='k-metrics-graph', style={'height': '85vh', 'display': 'none'}) 
            )
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
])

# 逻辑回调 1：动态聚类计算
@app.callback(
    [Output('galaxy-graph', 'figure'),
     Output('cluster-status', 'children')],
    [Input('cluster-btn', 'n_clicks')],
    [State('k-input', 'value')]
)
def update_clustering(n_clicks, k):
    """
    当用户点击聚类按钮时，根据当前 3D 坐标实时进行 K-Means 聚类并重绘。
    """
    if n_clicks == 0:
        return dash.no_update, ""
    
    print(f"正在对 3D 映射空间执行 K-Means (K={k})...")
    start_t = time.time()
    
    current_coords = df[['x', 'y', 'z']].values
    # n_init='auto' 是 sklearn 新版本推荐的设置
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(current_coords)
    
    new_fig = generate_scatter_figure(df, labels)
    msg = f"聚类成功 (K={k})！由于数据已降维，计算耗时仅为 {time.time()-start_t:.2f}s"
    print(msg)

    return new_fig, msg

# 逻辑回调 2：保存聚类结果到本地文件夹
@app.callback(
    Output('cluster-status', 'children', allow_duplicate=True),
    Input('save-btn', 'n_clicks'),
    State('k-input', 'value'),
    prevent_initial_call=True
)
def save_clustering_result(n_clicks, k):
    """
    当点击保存按钮时：
    1. 重新计算一次 K-Means 确保标签与当前设置同步。
    2. 创建“结果导出”文件夹。
    3. 按 Cluster ID 将数据（运动员、文件名、帧号）存入对应的 CSV。
    """
    if n_clicks == 0:
        return dash.no_update
    
    print(f"正在导出结果至文件夹 (K={k})...")
    
    # 1. 计算标签
    current_coords = df[['x', 'y', 'z']].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(current_coords)
    
    # 2. 准备导出目录
    export_dir = "output/结果导出"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # 3. 按类保存
    for cluster_id in range(k):
        # 筛选属于该类的行
        mask = (labels == cluster_id)
        cluster_df = df[mask][['skater', 'file', 'frame']]
        
        # 保存为 CSV
        file_path = os.path.join(export_dir, f"{cluster_id}.csv")
        cluster_df.to_csv(file_path, index=False)
        
    return f"✔️ 已成功导出 {k} 个分类文件至“{export_dir}”文件夹。"

# 逻辑回调 3：K 值评估分析
@app.callback(
    [Output('k-metrics-graph', 'figure'),
     Output('k-metrics-graph', 'style'),
     Output('pose-preview', 'style', allow_duplicate=True)], # 允许重复输出以控制显隐
    Input('analyze-btn', 'n_clicks'),
    prevent_initial_call=True
)
def analyze_k_value(n_clicks):
    """
    点击分析按钮时：
    1. 显示 K 值分析图 (block)。
    2. 隐藏骨架预览图 (none)。
    3. 执行全量计算并绘图。
    """
    if n_clicks == 0:
        return dash.no_update, {'display': 'none'}, {'display': 'block'}
    
    print("开始分析 K 值趋势 (这可能需要几分钟)...")
    coords = df[['x', 'y', 'z']].values
    
    k_range = range(8, 51)
    inertias = []
    silhouettes = []
    
    for k in k_range:
        # 计算 K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(coords)
        
        # 记录指标
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(coords, labels)) # 全量计算
        print(f"完成 K={k} 的计算...")
        
    # 创建双坐标轴图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Inertia (肘部法则)
    fig.add_trace(
        go.Scatter(x=list(k_range), y=inertias, name="Inertia (肘部指标)",
                   mode='lines+markers', line=dict(color='blue')),
        secondary_y=False
    )

    # Trace 2: Silhouette Score (轮廓系数)
    fig.add_trace(
        go.Scatter(x=list(k_range), y=silhouettes, name="Silhouette (轮廓系数)",
                   mode='lines+markers', line=dict(color='red')),
        secondary_y=True
    )

    # 布局设置
    fig.update_layout(
        title_text="最佳 K 值评估分析 (K=8~50)",
        hovermode="x unified" 
    )
    
    fig.update_xaxes(title_text="聚类数量 (K)")
    fig.update_yaxes(title_text="<b>Inertia</b> (越小越好)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Silhouette Score</b> (越大越好)", secondary_y=True)

    # 返回：Graph Figure, Graph Style (show), Pose Style (hide)
    return fig, {'height': '85vh', 'display': 'block'}, {'display': 'none'}

# 逻辑回调 4：联动显示 3D 骨骼预览
@app.callback(
    [Output('pose-preview', 'figure'),
     Output('pose-preview', 'style', allow_duplicate=True),
     Output('k-metrics-graph', 'style', allow_duplicate=True)],
    Input('galaxy-graph', 'hoverData'),
    prevent_initial_call=True
)
def update_pose_preview(hoverData):
    """
    当鼠标悬停星系图时：
    1. 自动切回骨架预览图 (block)。
    2. 隐藏 K 值分析图 (none)。
    同时加载对应的原始骨架文件、执行对齐并绘制。
    """
    if hoverData is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # 提取悬停点在 DataFrame 中的原始索引
    point_data = hoverData['points'][0]
    idx = point_data.get('customdata', point_data.get('pointNumber'))
    
    if idx is None:
        # 即使没有找到数据，也尝试切换回骨架视图
        return dash.no_update, {'height': '85vh', 'display': 'block'}, {'display': 'none'}

    target_info = df.iloc[idx]
    
    try:
        # 按需加载对应文件的特定帧
        pose_data = np.load(target_info['path'])
        frame_idx = int(target_info['frame'])
        raw_pose = pose_data[frame_idx]
        
        # 对预览骨架同样执行“对齐”操作
        aligned_pose = align_skeleton(raw_pose) 
        
        # 渲染关节点（编号 0-16）
        joints_scatter = go.Scatter3d(
            x=aligned_pose[:, 0], y=aligned_pose[:, 1], z=aligned_pose[:, 2],
            mode='markers+text',
            marker=dict(size=2, color='black'),
            text=[str(i) for i in range(17)],
            textposition="top center",
            name='关节点'
        )
        
        # 渲染人体拓扑连线
        bones_lines = []
        for start_j, end_j in H36M_CONNECTIONS:
            x = [aligned_pose[start_j, 0], aligned_pose[end_j, 0], None]
            y = [aligned_pose[start_j, 1], aligned_pose[end_j, 1], None]
            z = [aligned_pose[start_j, 2], aligned_pose[end_j, 2], None]
            bones_lines.append(go.Scatter3d(
                x=x, y=y, z=z, mode='lines', 
                line=dict(color='red', width=4), showlegend=False
            ))
            
        fig = go.Figure(data=[joints_scatter] + bones_lines)
        
        # 锁定预览坐标轴视角
        ax_range = 1000
        fig.update_layout(
            title=f"实时预览 | 运动员: {target_info['skater']} | 文件: {target_info['file']} | 帧: {frame_idx}",
            scene=dict(
                xaxis=dict(range=[-ax_range, ax_range], visible=False),
                yaxis=dict(range=[-ax_range, ax_range], visible=False),
                zaxis=dict(range=[-ax_range, ax_range], visible=False),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=50) 
        )
        # 返回三个 Output：Figure, Pose Style (显示), Metrics Style (隐藏)
        return fig, {'height': '85vh', 'display': 'block'}, {'display': 'none'}
        
    except Exception as err:
        print(f"预览加载失败: {err}")
        return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    print("应用即将启动，正在通过本地服务器加载...")
    print("提示：请在浏览器中访问 http://127.0.0.1:8050 查看可视化界面。")
    # 设置 use_reloader=False 避免在某些 IDE 环境下出现两次计算进程
    app.run(debug=True, use_reloader=False)
