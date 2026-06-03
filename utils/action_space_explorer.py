import os
import sys
import time
import json
import numpy as np

# 将项目根目录加入到sys.path，以便引入 npy_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npy_loader import normalize_skeleton
from skeleton_npz_loader import load_pose_sequence, load_all_skeleton_npz_files
from sklearn.metrics import pairwise_distances_argmin_min


def _configure_stdio():
    """Windows 默认 GBK 控制台无法输出 emoji，启动时尽量切到 UTF-8。"""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_stdio()


def safe_print(*args, **kwargs):
    """避免 Windows GBK 控制台因 emoji/特殊字符导致 UnicodeEncodeError。"""
    end = kwargs.pop("end", "\n")
    flush = kwargs.pop("flush", False)
    file = kwargs.pop("file", sys.stdout)
    text = " ".join(str(arg) for arg in args)
    payload = (text + end).encode("utf-8", errors="replace")
    buf = getattr(file, "buffer", None)
    if buf is not None:
        buf.write(payload)
        if flush:
            buf.flush()
        return
    try:
        file.write(text + end)
        if flush:
            file.flush()
    except UnicodeEncodeError:
        file.write(payload.decode("utf-8", errors="replace"))
        if flush:
            file.flush()

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
ORIGINAL84_CONNECTIONS = [
    (0, 4), (4, 9), (9, 15), (15, 18), (18, 19), (19, 21), (21, 20), (20, 18), (15, 19),
    (0, 22), (22, 27), (27, 33), (33, 36), (36, 37), (37, 39), (39, 38), (38, 36), (33, 37),
    (0, 40), (40, 41), (40, 45), (45, 46), (45, 48), (48, 52),
    (45, 53), (53, 58), (58, 62), (62, 65), (65, 66), (65, 67),
    (45, 68), (68, 73), (73, 77), (77, 80), (80, 81), (80, 82),
]
RIG_MODE = os.environ.get("ACTION_EXPLORER_RIG_MODE", "h36m17")
DATA_FORMAT = os.environ.get("ACTION_EXPLORER_DATA_FORMAT", "npy")
if DATA_FORMAT not in ("npy", "skeleton_npz"):
    DATA_FORMAT = "npy"
_default_data_root = (
    "./data/skeleton" if DATA_FORMAT == "skeleton_npz"
    else ("./data/npy_original84" if RIG_MODE == "original84" else "./data/npy")
)
DATA_ROOT = os.environ.get("ACTION_EXPLORER_DATA_ROOT", _default_data_root)
RUN_TSNE = os.environ.get("ACTION_EXPLORER_RUN_TSNE", "1") != "0"
_max_frames_env = os.environ.get("ACTION_EXPLORER_MAX_FRAMES", "").strip()
MAX_FRAMES = int(_max_frames_env) if _max_frames_env.isdigit() and int(_max_frames_env) > 0 else None

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
    
    # 增加归一化处理，约束同一动作在绝对空间内的骨骼长度比例一致
    aligned = normalize_skeleton(aligned)
    
    return aligned

def align_original84_frame(frame):
    """
    ORIGINAL 84 marker 专用预处理：
    1. 使用 PELVIS(0) 中心化。
    2. 使用 RASI(4) 与 LASI(22) 的骨盆横轴做朝向对齐。
    3. 使用 PELVIS(0) -> CLAV(45) 做整体尺度归一化。
    """
    frame = np.asarray(frame, dtype=np.float64)
    hip = frame[0]
    centered = frame - hip

    v_hip = centered[22] - centered[4]
    v_hip_xy = np.array([v_hip[0], v_hip[1], 0.0])
    norm = np.linalg.norm(v_hip_xy)
    if norm >= 1e-6:
        v_hip_norm = v_hip_xy / norm
        theta = np.arctan2(v_hip_norm[1], v_hip_norm[0])
        c, s = np.cos(-theta), np.sin(-theta)
        rotation = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        centered = centered @ rotation.T

    torso_len = np.linalg.norm(centered[45] - centered[0])
    if torso_len > 1e-6:
        centered = centered * (510.6 / torso_len)
    return centered


def align_frame_by_mode(frame, rig_mode):
    if rig_mode == "original84":
        return align_original84_frame(frame)
    return align_skeleton(frame)


def pose_connections_by_mode(rig_mode):
    if rig_mode == "original84":
        return ORIGINAL84_CONNECTIONS
    return H36M_CONNECTIONS


def pose_labels_by_mode(joint_count, rig_mode):
    if rig_mode == "original84":
        return [str(i) for i in range(joint_count)]
    return [str(i) for i in range(min(17, joint_count))]

def load_and_process_data(root_dir=DATA_ROOT, rig_mode=RIG_MODE, data_format=DATA_FORMAT):
    """
    加载姿态数据，执行对齐，并使用 t-SNE 进行降维映射。
    支持：
    - 单个 .npy / skeleton .npz 文件
    - 包含多个数据文件的目录
    """
    all_frames = []
    metadata = []  # 存储每帧的元数据：文件名、帧号、完整路径、运动员ID
    
    safe_print(
        f"正在加载并对齐数据... data_format={data_format}, rig_mode={rig_mode}, root_dir={root_dir}"
    )
    if data_format == "skeleton_npz":
        safe_print("警告: skeleton_npz 全量加载可能占用大量内存与耗时。")
    if MAX_FRAMES is not None:
        safe_print(f"已启用帧数上限 ACTION_EXPLORER_MAX_FRAMES={MAX_FRAMES}")
    files_processed = 0
    start_time = time.time()
    frames_loaded = 0
    
    def process_single_file(path):
        """处理单个姿态文件"""
        nonlocal files_processed, frames_loaded
        if MAX_FRAMES is not None and frames_loaded >= MAX_FRAMES:
            return
        data = load_pose_sequence(path)
        filename = os.path.basename(path)
        
        for i in range(data.shape[0]):
            if MAX_FRAMES is not None and frames_loaded >= MAX_FRAMES:
                break
            frame = data[i]
            aligned = align_frame_by_mode(frame, rig_mode)
            
            if data_format == "skeleton_npz":
                skater_name = "skeleton"
            else:
                path_parts = os.path.normpath(path).split(os.sep)
                try:
                    root_name = 'npy_original84' if rig_mode == "original84" else 'npy'
                    npy_index = path_parts.index(root_name)
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
            frames_loaded += 1
        files_processed += 1
    
    file_paths = []
    if data_format == "skeleton_npz":
        try:
            file_paths = load_all_skeleton_npz_files(root_dir)
        except FileNotFoundError:
            safe_print(f"错误：路径不存在或不是有效的 skeleton .npz 文件/目录: {root_dir}")
            return pd.DataFrame(), np.array([])
    elif os.path.isfile(root_dir) and root_dir.endswith('.npy'):
        file_paths = [root_dir]
    elif os.path.isdir(root_dir):
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.npy'):
                    file_paths.append(os.path.join(root, f))
    else:
        safe_print(f"错误：路径不存在或不是有效的 .npy 文件/目录: {root_dir}")
        return pd.DataFrame(), np.array([])

    if not file_paths:
        safe_print(f"错误：在 {root_dir} 下未找到数据文件。")
        return pd.DataFrame(), np.array([])

    safe_print(f"扫描到 {len(file_paths)} 个数据文件，开始逐文件加载...")
    for path in file_paths:
        if MAX_FRAMES is not None and frames_loaded >= MAX_FRAMES:
            break
        process_single_file(path)

    safe_print(f"数据加载完成。处理了 {files_processed} 个文件，共 {len(all_frames)} 帧。耗时: {time.time()-start_time:.2f}s")
    
    X = np.array(all_frames)
    
    # 降维处理第一步：PCA 降噪
    # 作用：自动压缩维度，并保留 95% 以上的核心方差，同时极大地加速和去噪。
    safe_print("正在执行 PCA 降噪...")
    pca_components = min(0.95, X.shape[0] - 1) if X.shape[0] <= 2 else 0.95
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)
    safe_print(f"PCA 已将维度压缩至 {X_pca.shape[1]} 维。")
    
    # 降维处理第二步：t-SNE 降维至 3D 空间
    # 作用：捕捉高维特征中的非线性结构（聚类团块），使结果利于 3D 可视化。
    if RUN_TSNE and X_pca.shape[0] > 3:
        safe_print("正在执行 t-SNE 降维 (这可能需要几分钟重新计算)...")
        perplexity = min(80, max(2, X_pca.shape[0] - 1))
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            max_iter=1500,
            init='pca',
            learning_rate='auto',
            early_exaggeration=20,
            verbose=2)
        X_embedded = tsne.fit_transform(X_pca)
        safe_print("t-SNE 降维计算完成。")
    else:
        safe_print("跳过 t-SNE，使用 PCA 前 3 维作为预览坐标。")
        X_embedded = np.zeros((X_pca.shape[0], 3))
        cols = min(3, X_pca.shape[1])
        X_embedded[:, :cols] = X_pca[:, :cols]
    
    # 将降维后的坐标添加到元数据 DataFrame
    df = pd.DataFrame(metadata)
    df['x'] = X_embedded[:, 0]
    df['y'] = X_embedded[:, 1]
    df['z'] = X_embedded[:, 2]
    
    # 补充返回 PCA 高维特征矩阵，供给下游 KMeans 切分使用以提拉精度
    return df, X_pca

# --- 2. 缓存管理 ---
# 为了避免每次启动都耗费几分钟计算特征转换，我们将结果全量缓存
if DATA_FORMAT == "skeleton_npz":
    CACHE_SUFFIX = "_skeleton_npz"
elif RIG_MODE == "original84":
    CACHE_SUFFIX = "_original84"
else:
    CACHE_SUFFIX = ""
CACHE_FILE = f'output/action_space_cache{CACHE_SUFFIX}.csv'
PCA_CACHE_FILE = f'output/action_space_cache_pca{CACHE_SUFFIX}.npy'
TEMPLATE_FILE = f'output/action_templates{CACHE_SUFFIX}.json'
EXPORT_DIR = f"output/结果导出{CACHE_SUFFIX}"

# 确保输出目录存在
if not os.path.exists('output'):
    os.makedirs('output')

if os.path.exists(CACHE_FILE) and os.path.exists(PCA_CACHE_FILE):
    safe_print(f"检测到缓存文件 {CACHE_FILE} 与特征阵，正在快速加载...")
    df = pd.read_csv(CACHE_FILE)
    PCA_FEATURES = np.load(PCA_CACHE_FILE)
else:
    df, PCA_FEATURES = load_and_process_data(DATA_ROOT, RIG_MODE, DATA_FORMAT)
    # 首次计算后自动导出缓存
    df.to_csv(CACHE_FILE, index=False)
    np.save(PCA_CACHE_FILE, PCA_FEATURES)
    safe_print(f"分析结果及其高维抽象已保存至 output/ 目录下。")

N_SAMPLES = len(df)
DEFAULT_K = min(500, max(2, N_SAMPLES))
if N_SAMPLES < 500:
    safe_print(
        f"提示: 当前加载 {N_SAMPLES} 帧。聚类数 K 不可超过样本数；"
        f"默认 K={DEFAULT_K}。若样本过少，请删除 {CACHE_FILE} 与 {PCA_CACHE_FILE} 后重新加载数据。"
    )

# --- 3. Dash 应用布局与交互 ---


def _resolve_k_clusters(k, n_samples: int) -> tuple[int | None, str | None]:
    """
    将用户输入的 K 约束到 [2, n_samples]，返回 (有效K, 提示信息)。
    无法聚类时返回 (None, 错误信息)。
    """
    if n_samples < 2:
        return None, f"当前仅有 {n_samples} 帧，至少需要 2 帧才能执行 K-Means。"

    try:
        k_int = int(k)
    except (TypeError, ValueError):
        k_int = min(500, n_samples)

    if k_int < 2:
        k_int = 2

    if k_int > n_samples:
        effective = n_samples
        return effective, (
            f"K 已从 {k_int} 自动调整为 {effective}（样本数 {n_samples}，K 不能大于样本数）。"
        )

    return k_int, None


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
        dcc.Input(id='k-input', type='number', value=DEFAULT_K, min=2, max=max(2, N_SAMPLES), step=1, style={'width': '60px', 'marginRight': '10px'}),
        html.Span(f"（已加载 {N_SAMPLES} 帧，K ≤ {N_SAMPLES}）", style={'marginRight': '10px', 'color': '#666', 'fontSize': '13px'}),
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

    effective_k, adjust_msg = _resolve_k_clusters(k, N_SAMPLES)
    if effective_k is None:
        return dash.no_update, adjust_msg

    safe_print(f"正在对 3D 映射空间执行 K-Means (K={effective_k})...")
    start_t = time.time()

    features_for_clustering = PCA_FEATURES
    kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features_for_clustering)

    new_fig = generate_scatter_figure(df, labels)
    msg = f"聚类成功 (K={effective_k})！计算耗时 {time.time()-start_t:.2f}s"
    if adjust_msg:
        msg = adjust_msg + " " + msg
    safe_print(msg)

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

    try:
        effective_k, adjust_msg = _resolve_k_clusters(k, N_SAMPLES)
        if effective_k is None:
            return adjust_msg

        safe_print(f"正在导出结果至文件夹 (K={effective_k})...")

        features_for_clustering = PCA_FEATURES
        kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_for_clustering)

        export_dir = EXPORT_DIR
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        for cluster_id in range(effective_k):
            mask = (labels == cluster_id)
            cluster_df = df[mask][['skater', 'file', 'frame']]
            file_path = os.path.join(export_dir, f"{cluster_id}.csv")
            cluster_df.to_csv(file_path, index=False)

        safe_print("正在构建标准化动作模板核心骨架...")
        template_file = TEMPLATE_FILE
        centers = kmeans.cluster_centers_
        closest_indices, _ = pairwise_distances_argmin_min(centers, features_for_clustering)

        templates_data = {}
        for c_id, idx in enumerate(closest_indices):
            target_info = df.iloc[idx]
            pose_data = load_pose_sequence(target_info['path'])
            frame_idx = int(target_info['frame'])
            raw_pose = pose_data[frame_idx]
            aligned_pose = align_frame_by_mode(raw_pose, RIG_MODE)

            templates_data[str(c_id)] = {
                "cluster_id": c_id,
                "label": str(c_id),
                "rig_mode": RIG_MODE,
                "joint_count": int(aligned_pose.shape[0]),
                "source_file": str(target_info['file']),
                "frame_idx": frame_idx,
                "source_path": str(target_info['path']),
                "skeleton": aligned_pose.tolist()
            }

        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(templates_data, f, ensure_ascii=False, indent=4)

        safe_print(f"[OK] template saved: {template_file}")
        result = (
            f"已成功导出 {effective_k} 个分类文件至“{export_dir}”文件夹，"
            f"并创建模板动作字典库：{template_file}"
        )
        if adjust_msg:
            result = adjust_msg + " " + result
        return result
    except Exception as e:
        safe_print(f"[ERROR] save failed: {type(e).__name__}: {e!r}")
        return f"保存失败: {type(e).__name__}: {e}"

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
    
    safe_print("开始分析 K 值趋势 (这可能需要几分钟)...")
    features_for_clustering = PCA_FEATURES

    if N_SAMPLES < 3:
        empty = go.Figure()
        empty.update_layout(title=f"样本仅 {N_SAMPLES} 帧，无法进行 K=8~50 趋势分析（至少需要 3 帧）")
        return empty, {'height': '85vh', 'display': 'block'}, {'display': 'none'}

    k_min = 2 if N_SAMPLES < 8 else 8
    k_max = min(50, N_SAMPLES - 1)
    if k_max < k_min:
        empty = go.Figure()
        empty.update_layout(title=f"样本数 {N_SAMPLES} 不足，K 趋势分析需要至少 {k_min + 1} 帧")
        return empty, {'height': '85vh', 'display': 'block'}, {'display': 'none'}

    k_range = range(k_min, k_max + 1)
    inertias = []
    silhouettes = []
    
    for k in k_range:
        # 计算 K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features_for_clustering)
        
        # 记录指标
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(features_for_clustering, labels)) # 全量高维计算
        safe_print(f"完成 K={k} 的计算...")
        
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
        pose_data = load_pose_sequence(target_info['path'])
        frame_idx = int(target_info['frame'])
        raw_pose = pose_data[frame_idx]
        
        # 对预览骨架同样执行对应模式的“对齐”操作
        aligned_pose = align_frame_by_mode(raw_pose, RIG_MODE)
        connections = pose_connections_by_mode(RIG_MODE)
        labels = pose_labels_by_mode(aligned_pose.shape[0], RIG_MODE)
        
        # 渲染关节点
        joints_scatter = go.Scatter3d(
            x=aligned_pose[:, 0], y=aligned_pose[:, 1], z=aligned_pose[:, 2],
            mode='markers+text',
            marker=dict(size=2 if RIG_MODE == "original84" else 4, color='black'),
            text=labels,
            textposition="top center",
            name='关节点'
        )
        
        # 渲染人体拓扑连线
        bones_lines = []
        for start_j, end_j in connections:
            if start_j >= aligned_pose.shape[0] or end_j >= aligned_pose.shape[0]:
                continue
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
        safe_print(f"预览加载失败: {err}")
        return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    safe_print("应用即将启动，正在通过本地服务器加载...")
    safe_print("提示：请在浏览器中访问 http://127.0.0.1:8050 查看可视化界面。")
    # 设置 use_reloader=False 避免在某些 IDE 环境下出现两次计算进程
    app.run(debug=True, use_reloader=False)
