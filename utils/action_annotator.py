import os
import sys
import json
import base64
from pathlib import Path
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go

try:
    import cv2
except ImportError:
    cv2 = None

# 导入项目中通用的基础算法，用以进行测试帧与历史帧一样的环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npy_loader import normalize_skeleton
from skeleton_npz_loader import load_pose_sequence

# 定义画图连线规范
H36M_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
    (12, 13), (14, 15), (15, 16)
]
FOOT_MARKERS = ["RHEL", "R_Toe", "RBAM", "RMT5", "LHEL", "L_Toe", "LBAM", "LMT5"]
FOOT_CONNECTIONS = [
    (3, "RHEL"), ("RHEL", "R_Toe"), ("R_Toe", "RMT5"), ("RMT5", "RBAM"), ("RBAM", "RHEL"), (3, "R_Toe"),
    (6, "LHEL"), ("LHEL", "L_Toe"), ("L_Toe", "LMT5"), ("LMT5", "LBAM"), ("LBAM", "LHEL"), (6, "L_Toe"),
]

# ======= 动作字典常量 =======
JUMP_TYPES = [
    {"label": "Axel (A)", "value": "A"},
    {"label": "Toeloop (T)", "value": "T"},
    {"label": "Loop (Lo)", "value": "Lo"},
    {"label": "Salchow (S)", "value": "S"},
    {"label": "Flip (F)", "value": "F"},
    {"label": "Lutz (Lz)", "value": "Lz"},
    {"label": "Combination (Co)", "value": "Co"}
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
    {"label": "H1: 头部左旋转", "value": "H1"},
    {"label": "H2: 头部右旋转", "value": "H2"},
    {"label": "H3: 头上仰", "value": "H3"},
    {"label": "H4: 头下低", "value": "H4"},
    {"label": "H5: 头部直视前方", "value": "H5"},
    
    {"label": "T1: 躯干左旋转", "value": "T1"},
    {"label": "T2: 躯干右旋转", "value": "T2"},
    {"label": "T3: 躯干前倾", "value": "T3"},
    {"label": "T4: 躯干后仰", "value": "T4"},
    {"label": "T5: 躯干左倾斜", "value": "T5"},
    {"label": "T6: 躯干右倾斜", "value": "T6"},
    {"label": "T7: 躯干直立", "value": "T7"},
    
    {"label": "L1: 左上臂前屈", "value": "L1"},
    {"label": "L2: 左上臂外展", "value": "L2"},
    {"label": "L3: 左上臂后伸", "value": "L3"},
    {"label": "L4: 左上臂内收", "value": "L4"},
    {"label": "L5: 手掌打开", "value": "L5"},
    {"label": "L6: 手掌握拳", "value": "L6"},
    {"label": "L7: 左臂下垂", "value": "L7"},
    
    {"label": "R1: 右上臂前屈", "value": "R1"},
    {"label": "R2: 右上臂外展", "value": "R2"},
    {"label": "R3: 右上臂后伸", "value": "R3"},
    {"label": "R4: 右上臂内收", "value": "R4"},
    {"label": "R5: 手掌打开", "value": "R5"},
    {"label": "R6: 手掌握拳", "value": "R6"},
    {"label": "R7: 右臂下垂", "value": "R7"},
    
    {"label": "l1: 左髋前屈", "value": "l1"},
    {"label": "l2: 左髋外展", "value": "l2"},
    {"label": "l3: 左髋后伸", "value": "l3"},
    {"label": "l4: 左髋内收", "value": "l4"},
    {"label": "l5: 左膝弯曲", "value": "l5"},
    {"label": "l6: 点冰（脚尖下压）", "value": "l6"},
    {"label": "l7: 勾脚尖", "value": "l7"},
    {"label": "l8: 左腿直立", "value": "l8"},
    
    {"label": "r1: 右髋前屈", "value": "r1"},
    {"label": "r2: 右髋外展", "value": "r2"},
    {"label": "r3: 右髋后伸", "value": "r3"},
    {"label": "r4: 右髋内收", "value": "r4"},
    {"label": "r5: 右膝弯曲", "value": "r5"},
    {"label": "r6: 点冰（脚尖下压）", "value": "r6"},
    {"label": "r7: 勾脚尖", "value": "r7"},
    {"label": "r8: 右腿直立", "value": "r8"}
]
# ===========================

DB_PATH = os.environ.get("ACTION_TEMPLATE_FILE", "output/action_templates.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = PROJECT_ROOT / "data" / "video"
JSON_ROOT = PROJECT_ROOT / "data" / "json"
CAMERA_IDS = [f"cam_{idx}" for idx in range(1, 13)]
VIDEO_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        b"""
        <svg xmlns="http://www.w3.org/2000/svg" width="960" height="540">
            <rect width="100%" height="100%" fill="#f1f3f5"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
                  font-family="Arial" font-size="28" fill="#868e96">
                No video frame
            </text>
        </svg>
        """
    ).decode("ascii")
)

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


def _normalize_candidate(path_like):
    path = Path(str(path_like).replace("\\", os.sep).replace("/", os.sep))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


SKELETON_SKIP_VIDEO_MSG = "Skeleton NPZ 数据无配套视频，已跳过视频匹配。"


def is_skeleton_source(data_or_path) -> bool:
    """判断模板/路径是否来自 skeleton NPZ（无配套 video/json）。"""
    if isinstance(data_or_path, dict):
        if str(data_or_path.get("data_format", "")).lower() == "skeleton_npz":
            return True
        rig_mode = str(data_or_path.get("rig_mode", "")).lower()
        if rig_mode in ("skeleton_npz", "skeleton"):
            return True
        source_path = data_or_path.get("source_path") or data_or_path.get("source_file")
        if source_path:
            return is_skeleton_source(source_path)
        return False

    path_str = str(data_or_path or "").strip()
    if not path_str:
        return False
    normalized = path_str.replace("\\", "/").lower()
    if normalized.endswith(".npz"):
        return True
    return "skeleton" in normalized


def build_skeleton_video_response(source_path=None, source_file=None, frame_idx=0):
    """skeleton 模板：禁用视频控件，返回占位状态。"""
    controls = build_video_control_props(None, 0)
    return {
        "camera_options": [{"label": "无需视频", "value": "none"}],
        "camera_value": "none",
        "video_store": {
            "skip_video": True,
            "source_path": source_path,
            "source_file": source_file,
            "frame_idx": int(frame_idx or 0),
            "offset": 0,
            "camera": "none",
            "metadata": {},
        },
        "status": SKELETON_SKIP_VIDEO_MSG,
        "frame_src": VIDEO_PLACEHOLDER,
        "controls": controls,
    }


def _skater_from_source_path(source_path):
    if not source_path:
        return None

    npy_path = _normalize_candidate(source_path)
    parts = list(npy_path.parts)
    lower_parts = [p.lower() for p in parts]
    if "npy" not in lower_parts:
        return None

    idx = lower_parts.index("npy")
    rel_parts = parts[idx + 1:]
    if not rel_parts:
        return None

    skater = rel_parts[0]
    return skater[:1].lower() + skater[1:]


def _source_video_name(source_path=None, source_file=None):
    if source_file and source_file != '未知':
        return Path(str(source_file)).with_suffix(".mp4").name
    if source_path:
        return Path(str(source_path)).with_suffix(".mp4").name
    return None


def discover_camera_options(source_path=None, source_file=None):
    skater = _skater_from_source_path(source_path)
    video_name = _source_video_name(source_path, source_file)
    options = []

    for camera in CAMERA_IDS:
        label = camera
        if skater and video_name:
            candidate = VIDEO_ROOT / skater / camera / video_name
            if candidate.exists():
                label = f"{camera} (可用)"
        options.append({"label": label, "value": camera})

    return options


def get_npy_video_offset(source_path):
    """
    解析对应的 JSON 文件，计算该 npy 姿态在视频中的起始帧偏移量。
    算法参考原始数据集 format.py 的获取公共区间逻辑。
    """
    if not source_path: return 0
    
    # 构建 JSON 路径
    p = Path(source_path)
    parts = list(p.parts)
    lower_parts = [x.lower() for x in parts]
    if "npy" in lower_parts:
        idx = lower_parts.index("npy")
        parts[idx] = "json"
    json_path = Path(*parts).with_suffix(".json")
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def get_main_range(parts):
        ranges = [(p['Range']['Start'], p['Range']['End']) for p in parts]
        lengths = [e - s for s, e in ranges]
        idx = int(np.argmax(lengths))
        return ranges[idx][0] - 1, ranges[idx][1] - 1

    markers = data['Markers']
    s0, _ = get_main_range(markers[0]['Parts'])
    t_start = s0
    for m in markers:
        ts, _ = get_main_range(m['Parts'])
        t_start = max(t_start, ts)
    
    return t_start


def resolve_json_path(source_path):
    if not source_path:
        return None

    parts = list(Path(str(source_path).replace("\\", os.sep).replace("/", os.sep)).parts)
    lower_parts = [x.lower() for x in parts]
    if "npy" not in lower_parts:
        return None

    parts[lower_parts.index("npy")] = "json"
    json_path = Path(*parts).with_suffix(".json")
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    return json_path


def _main_marker_part(marker):
    parts = marker.get("Parts", [])
    if not parts:
        return None, 0

    def part_len(part):
        frame_range = part.get("Range", {})
        return frame_range.get("End", 0) - frame_range.get("Start", 0)

    main_part = max(parts, key=part_len)
    start = int(main_part.get("Range", {}).get("Start", 1)) - 1
    return main_part, start


def _align_extra_points(points, raw_pose, aligned_pose):
    hip = raw_pose[0]
    centered = points - hip
    v_hip = raw_pose[4] - raw_pose[1]
    v_hip_xy = np.array([v_hip[0], v_hip[1], 0])
    norm = np.linalg.norm(v_hip_xy)
    if norm >= 1e-6:
        v_hip_norm = v_hip_xy / norm
        theta = np.arctan2(v_hip_norm[1], v_hip_norm[0])
        c, s = np.cos(-theta), np.sin(-theta)
        rotation = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        centered = centered @ rotation.T

    raw_len = np.linalg.norm(raw_pose[9] - raw_pose[0])
    aligned_len = np.linalg.norm(aligned_pose[9] - aligned_pose[0])
    if raw_len > 1e-6 and aligned_len > 1e-6:
        centered = centered * (aligned_len / raw_len)
    return centered


def load_foot_points(source_path, frame_idx, aligned_pose):
    json_path = resolve_json_path(source_path)
    if not json_path or not json_path.exists():
        return {}

    try:
        npy_path = _normalize_candidate(source_path)
        raw_data = np.load(npy_path)
        safe_frame = max(0, min(int(frame_idx or 0), raw_data.shape[0] - 1))
        raw_pose = raw_data[safe_frame]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_frame = get_npy_video_offset(source_path) + safe_frame
        marker_by_name = {marker.get("Name"): marker for marker in data.get("Markers", [])}
        names = []
        points = []
        for name in FOOT_MARKERS:
            marker = marker_by_name.get(name)
            if not marker:
                continue
            part, part_start = _main_marker_part(marker)
            if not part:
                continue
            values = part.get("Values", [])
            value_idx = json_frame - part_start
            if value_idx < 0 or value_idx >= len(values):
                continue
            names.append(name)
            points.append(values[value_idx][:3])

        if not points:
            return {}

        aligned_points = _align_extra_points(np.asarray(points, dtype=float), raw_pose, aligned_pose)
        return {name: point for name, point in zip(names, aligned_points)}
    except Exception as e:
        print(f"读取脚部 marker 失败: {e}")
        return {}


def resolve_video_path(source_path=None, source_file=None, camera="cam_1"):
    candidates = []
    skater = _skater_from_source_path(source_path)
    video_name = _source_video_name(source_path, source_file)

    if skater and video_name and camera:
        candidates.append(VIDEO_ROOT / skater / camera / video_name)

    if source_path:
        npy_path = _normalize_candidate(source_path)
        parts = list(npy_path.parts)
        lower_parts = [p.lower() for p in parts]
        if "npy" in lower_parts:
            idx = lower_parts.index("npy")
            rel_parts = parts[idx + 1:]
            if rel_parts:
                mapped_parts = list(rel_parts)
                mapped_parts[0] = mapped_parts[0][:1].lower() + mapped_parts[0][1:]
                candidates.append((VIDEO_ROOT / Path(*mapped_parts)).with_suffix(".mp4"))

                legacy_parts = parts[:]
                legacy_parts[idx] = "video"
                candidates.append(Path(*legacy_parts).with_suffix(".mp4"))

    if source_file:
        source_name = Path(str(source_file)).with_suffix(".mp4").name
        for root, _, file_names in os.walk(VIDEO_ROOT):
            for file_name in file_names:
                if file_name.lower() == source_name.lower():
                    candidates.append(Path(root) / file_name)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    return None


def get_video_metadata(video_path):
    if cv2 is None:
        return None, "未安装 opencv-python，无法读取视频帧。"
    if not video_path:
        return None, "未找到同源视频。"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"视频无法打开: {video_path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()

    if frame_count <= 0:
        return None, f"视频帧数读取失败: {video_path}"

    duration = frame_count / fps if fps > 0 else 0
    return {"path": str(video_path), "frame_count": frame_count, "fps": fps, "duration": duration}, None


def encode_video_frame(video_path, frame_idx):
    if cv2 is None:
        return VIDEO_PLACEHOLDER, "未安装 opencv-python，无法读取视频帧。"

    metadata, error = get_video_metadata(video_path)
    if error:
        return VIDEO_PLACEHOLDER, error

    max_idx = metadata["frame_count"] - 1
    safe_idx = max(0, min(int(frame_idx or 0), max_idx))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return VIDEO_PLACEHOLDER, f"读取视频第 {safe_idx} 帧失败。"

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return VIDEO_PLACEHOLDER, "视频帧编码失败。"

    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", None


def build_video_control_props(video_metadata, anchor_frame):
    if not video_metadata:
        return {
            "range_min": 0,
            "range_max": 1,
            "range_value": [0, 1],
            "range_marks": {0: '0', 1: '1'},
            "range_disabled": True,
            "frame_min": 0,
            "frame_max": 1,
            "frame_value": 0,
            "frame_marks": {0: '0', 1: '1'},
            "frame_disabled": True,
        }

    frame_count = video_metadata["frame_count"]
    max_frame = frame_count - 1
    safe_anchor = max(0, min(int(anchor_frame or 0), max_frame))
    
    return {
        "frame_min": 0,
        "frame_max": max_frame,
        "frame_value": safe_anchor,
        "frame_marks": {0: '0', safe_anchor: f"对齐帧 {safe_anchor}", max_frame: str(max_frame)},
        "frame_disabled": False,
    }

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
def create_pose_figure(aligned_pose, title="动作展示", color='red', extra_points=None):
    if aligned_pose is None or len(aligned_pose) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, scene=dict(aspectmode='cube'))
        return fig
        
    joints_scatter = go.Scatter3d(
        x=aligned_pose[:, 0], y=aligned_pose[:, 1], z=aligned_pose[:, 2],
        mode='markers+text',
        marker=dict(size=2, color='black'),
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

    extra_traces = []
    extra_points = extra_points or {}
    if extra_points:
        names = list(extra_points.keys())
        pts = np.asarray([extra_points[name] for name in names], dtype=float)
        extra_traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers', # 'markers+text'删去text脚步节点名称的介绍
            marker=dict(size=2, color='#ff922b'),
            text=names,
            textposition="top center",
            name='脚部 marker'
        ))
        for start_j, end_j in FOOT_CONNECTIONS:
            if isinstance(start_j, int):
                start_point = aligned_pose[start_j]
            else:
                start_point = extra_points.get(start_j)
            if isinstance(end_j, int):
                end_point = aligned_pose[end_j]
            else:
                end_point = extra_points.get(end_j)
            if start_point is None or end_point is None:
                continue
            extra_traces.append(go.Scatter3d(
                x=[start_point[0], end_point[0], None],
                y=[start_point[1], end_point[1], None],
                z=[start_point[2], end_point[2], None],
                mode='lines',
                line=dict(color='#ff922b', width=3),
                showlegend=False
            ))

    fig = go.Figure(data=[joints_scatter] + bones_lines + extra_traces)
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
                    dcc.Dropdown(id='dropdown-action-units', options=ACTION_UNITS, multi=True, placeholder="可选择多项组合...", closeOnSelect=False),
                    
                    html.Div(id='label-preview', style={'marginTop': '15px', 'padding': '10px', 'fontFamily': 'monospace', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                    
                    html.Button("💾 录入并更名", id='save-label-btn', n_clicks=0, style={'marginTop': '15px', 'padding': '10px 20px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '5px', 'fontWeight': 'bold'}),
                    html.Div(id='save-result-msg', style={'marginTop': '15px', 'color': 'blue'})
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxSizing': 'border-box'}),
                
                # 右侧：标注帧骨架与同源视频帧同列展示
                html.Div([
                    dcc.Store(id='template-video-store', data={}),
                    dcc.Graph(id='template-pose-graph', style={'height': '44vh', 'width': '100%'}),
                    html.Div([
                        html.H3("同源视频辅助帧", style={'textAlign': 'center', 'margin': '10px 0'}),
                        html.Label("选择视频视角", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            id='template-video-camera-dropdown',
                            options=[{"label": camera, "value": camera} for camera in CAMERA_IDS],
                            value='cam_1',
                            clearable=False,
                            style={'marginBottom': '8px'}
                        ),
                        html.Div(id='template-video-status', style={'whiteSpace': 'pre-wrap', 'color': '#495057', 'marginBottom': '8px'}),
                        html.Img(
                            id='template-video-frame',
                            src=VIDEO_PLACEHOLDER,
                            style={'width': '100%', 'height': '34vh', 'objectFit': 'contain', 'backgroundColor': '#111', 'borderRadius': '6px'}
                        ),
                        html.Div([
                            html.Label("手动调整视频帧 (当前已自动对齐)", style={'fontWeight': 'bold', 'display': 'block', 'marginTop': '10px'}),
                            dcc.Slider(
                                id='template-video-frame-slider',
                                min=0, max=1, step=1, value=0,
                                marks={0: '0', 1: '1'},
                                disabled=True,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 8px 8px 8px'})
                    ], style={'width': '100%', 'boxSizing': 'border-box', 'padding': '10px 16px 0 16px'})
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ]),

        # ====================== [板块 2] : 新品动作比令人判决 ======================
        dcc.Tab(label='🔬 单帧实时诊断与配型', children=[
            html.Div([
                html.H3("指派本地姿态数据源路径进行定级", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Input(id='test-file-path', type='text', placeholder="请输入 .npy 或 skeleton .npz 路径（例：./data/skeleton/0.npz）", style={'width': '50%', 'padding': '10px'}),
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
    Output('template-dropdown', 'options'),
    Input('template-dropdown', 'id') 
)
def initialize_dropdown_options(_):
    db = load_db()
    if not db:
        return []
    options = [{'label': f"簇 {cid} ({db[cid]['label']})", 'value': cid} 
               for cid in sorted(db.keys(), key=lambda x: int(x))]
    return options

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
    
    # 💡 改进：按 H, T, L, R, l, r 顺序排序
    if units:
        order = {'H': 0, 'T': 1, 'L': 2, 'R': 3, 'l': 4, 'r': 5}
        sorted_units = sorted(units, key=lambda x: (order.get(x[0], 99), int(x[1:]) if x[1:].isdigit() else 0))
        u_str = "[" + ",".join(sorted_units) + "]"
    else:
        u_str = "[]"
    
    preview_text = f"{j_str}-{s_str}-{t_str}-{u_str}"
    return f"🏷️ 预览编码格式: {preview_text}"

@app.callback(
    [Output('template-pose-graph', 'figure'),
     Output('template-info', 'children'),
     Output('dropdown-jump-type', 'value'),
     Output('dropdown-stage', 'value'),
     Output('radio-temporal-flag', 'value'),
     Output('dropdown-action-units', 'value'),
     Output('template-video-camera-dropdown', 'options'),
     Output('template-video-camera-dropdown', 'value'),
     Output('template-video-store', 'data'),
     Output('template-video-status', 'children'),
     Output('template-video-frame', 'src'),
     Output('template-video-frame-slider', 'min'),
     Output('template-video-frame-slider', 'max'),
     Output('template-video-frame-slider', 'value'),
     Output('template-video-frame-slider', 'marks'),
     Output('template-video-frame-slider', 'disabled')],
    Input('template-dropdown', 'value'),
    prevent_initial_call=False
)
def render_template(cluster_id):
    if cluster_id is None:
        controls = build_video_control_props(None, 0)
        return (
            dash.no_update, "无展示数据。", None, None, "S", [],
            discover_camera_options(), "cam_1", {}, "无展示数据。", VIDEO_PLACEHOLDER,
            controls["frame_min"], controls["frame_max"], controls["frame_value"], controls["frame_marks"], controls["frame_disabled"],
        )
    
    db = load_db()
    if cluster_id not in db:
        controls = build_video_control_props(None, 0)
        return (
            dash.no_update, "分类字典已损坏或缺失", None, None, "S", [],
            discover_camera_options(), "cam_1", {}, "分类字典已损坏或缺失。", VIDEO_PLACEHOLDER,
            controls["frame_min"], controls["frame_max"], controls["frame_value"], controls["frame_marks"], controls["frame_disabled"],
        )
        
    data = db[cluster_id]
    skeleton = np.array(data['skeleton'])
    source_path = data.get('source_path')
    source_file = data.get('source_file', '未知')
    frame_idx = int(data.get('frame_idx', 0) or 0)
    skip_video = is_skeleton_source(data)

    foot_points = {} if skip_video else load_foot_points(source_path, frame_idx, skeleton)
    fig = create_pose_figure(skeleton, title=f"选定模型 - ID {cluster_id}", color='deepskyblue', extra_points=foot_points)
    
    info_text = f"📍 模板标识: 【 {data.get('label', '未标注')} 】\n"
    info_text += f"📂 萃取来源: {source_file}\n"
    info_text += f"🎞️ 所在原帧: 第 {frame_idx} 帧\n"
    if skip_video:
        info_text += f"📹 视频: 无（Skeleton NPZ）\n"
    elif foot_points:
        info_text += f"🦶 脚部 marker: 已加载 {len(foot_points)} 个\n"
    
    metadata = data.get('metadata', {})
    j_val = metadata.get('jump_type', None)
    
    # --- 💡 极速标注改进：根据路径智能推断 ---
    if not j_val and source_path and not skip_video:
        path_str = str(source_path).lower()
        if 'axel' in path_str: j_val = "A"
        elif 'toeloop' in path_str: j_val = "T"
        elif 'loop' in path_str: j_val = "Lo"
        elif 'salchow' in path_str: j_val = "S"
        elif 'flip' in path_str: j_val = "F"
        elif 'lutz' in path_str: j_val = "Lz"
        elif 'comb' in path_str: j_val = "Co"
    # ------------------------------------

    s_val = metadata.get('stage', None)
    t_val = metadata.get('temporal_flag', "S")
    u_val = metadata.get('action_units', [])

    if skip_video:
        skel_video = build_skeleton_video_response(source_path, source_file, frame_idx)
        controls = skel_video["controls"]
        return (
            fig, info_text, j_val, s_val, t_val, u_val,
            skel_video["camera_options"], skel_video["camera_value"],
            skel_video["video_store"], skel_video["status"], skel_video["frame_src"],
            controls["frame_min"], controls["frame_max"], controls["frame_value"],
            controls["frame_marks"], controls["frame_disabled"],
        )

    camera_options = discover_camera_options(source_path, source_file)
    camera_value = metadata.get('camera', 'cam_1') # 💡 改进：优先从已保存的元数据中读取视角

    video_path = resolve_video_path(source_path, source_file, camera_value)
    video_metadata, video_error = get_video_metadata(video_path)
    
    # 获取 JSON 偏移量并对齐
    offset = get_npy_video_offset(source_path)
    absolute_video_frame = frame_idx + offset
    
    controls = build_video_control_props(video_metadata, absolute_video_frame)

    if video_metadata:
        frame_src, frame_error = encode_video_frame(video_metadata["path"], absolute_video_frame)
        fps_text = f"{video_metadata['fps']:.2f}" if video_metadata["fps"] else "未知"
        status = (
            f"视角: {camera_value} | 对齐偏移: {offset}\n"
            f"视频: {video_metadata['path']}\n"
            f"帧数: {video_metadata['frame_count']} | FPS: {fps_text} | 视频当前帧: {min(absolute_video_frame, video_metadata['frame_count'] - 1)}"
        )
        if frame_error:
            status += f"\n{frame_error}"
    else:
        frame_src = VIDEO_PLACEHOLDER
        status = video_error or "未找到同源视频。"
        if source_path:
            status += f"\n源姿态路径: {source_path}"
        status += f"\n当前视角: {camera_value}"

    video_store = {
        "source_path": source_path,
        "source_file": source_file,
        "frame_idx": frame_idx,
        "offset": offset,
        "camera": camera_value,
        "metadata": video_metadata or {},
    }
    
    return (
        fig, info_text, j_val, s_val, t_val, u_val, camera_options, camera_value, video_store, status, frame_src,
        controls["frame_min"], controls["frame_max"], controls["frame_value"], controls["frame_marks"], controls["frame_disabled"],
    )




@app.callback(
    [Output('template-video-store', 'data', allow_duplicate=True),
     Output('template-video-status', 'children', allow_duplicate=True),
     Output('template-video-frame', 'src', allow_duplicate=True),
     Output('template-video-frame-slider', 'min', allow_duplicate=True),
     Output('template-video-frame-slider', 'max', allow_duplicate=True),
     Output('template-video-frame-slider', 'value', allow_duplicate=True),
     Output('template-video-frame-slider', 'marks', allow_duplicate=True),
     Output('template-video-frame-slider', 'disabled', allow_duplicate=True)],
    Input('template-video-camera-dropdown', 'value'),
    [State('template-video-store', 'data'),
     State('template-video-frame-slider', 'value')],
    prevent_initial_call=True
)
def update_video_camera(camera, video_store, current_frame):
    if not video_store:
        controls = build_video_control_props(None, 0)
        return (
            {}, "无展示数据。", VIDEO_PLACEHOLDER,
            controls["frame_min"], controls["frame_max"], controls["frame_value"], controls["frame_marks"], controls["frame_disabled"],
        )

    if video_store.get("skip_video"):
        skel_video = build_skeleton_video_response(
            video_store.get("source_path"),
            video_store.get("source_file"),
            video_store.get("frame_idx", 0),
        )
        controls = skel_video["controls"]
        return (
            skel_video["video_store"], skel_video["status"], skel_video["frame_src"],
            controls["frame_min"], controls["frame_max"], controls["frame_value"],
            controls["frame_marks"], controls["frame_disabled"],
        )

    source_path = video_store.get("source_path")
    source_file = video_store.get("source_file")
    camera = camera or "cam_1"
    video_path = resolve_video_path(source_path, source_file, camera)
    video_metadata, video_error = get_video_metadata(video_path)
    offset = get_npy_video_offset(source_path)

    # 如果已经有滑块位置(绝对帧)，则保持；否则由 npy 参考帧 + offset 初始化
    if current_frame is not None:
        absolute_video_frame = int(current_frame)
    else:
        absolute_video_frame = int(video_store.get("frame_idx", 0) or 0) + offset
    
    controls = build_video_control_props(video_metadata, absolute_video_frame)

    if video_metadata:
        safe_frame = max(0, min(absolute_video_frame, video_metadata["frame_count"] - 1))
        frame_src, frame_error = encode_video_frame(video_metadata["path"], safe_frame)
        fps_text = f"{video_metadata['fps']:.2f}" if video_metadata["fps"] else "未知"
        status = (
            f"视角: {camera} | 对齐偏移: {offset}\n"
            f"视频: {video_metadata['path']}\n"
            f"帧数: {video_metadata['frame_count']} | FPS: {fps_text} | 视频当前帧: {safe_frame}"
        )
        if frame_error:
            status += f"\n{frame_error}"
    else:
        frame_src = VIDEO_PLACEHOLDER
        status = video_error or "未找到同源视频。"
        status += f"\n当前视角: {camera} | 偏移: {offset}"
        if source_path:
            status += f"\n源姿态路径: {source_path}"

    next_store = {
        "source_path": source_path,
        "source_file": source_file,
        "frame_idx": video_store.get("frame_idx", 0),
        "offset": offset,
        "camera": camera,
        "metadata": video_metadata or {},
    }
    return (
        next_store, status, frame_src,
        controls["frame_min"], controls["frame_max"], controls["frame_value"], controls["frame_marks"], controls["frame_disabled"],
    )


@app.callback(
    [Output('template-video-frame', 'src', allow_duplicate=True),
     Output('template-video-status', 'children', allow_duplicate=True)],
    Input('template-video-frame-slider', 'value'),
    State('template-video-store', 'data'),
    prevent_initial_call=True
)
def update_video_frame(frame_idx, video_metadata):
    if video_metadata and video_metadata.get("skip_video"):
        return VIDEO_PLACEHOLDER, SKELETON_SKIP_VIDEO_MSG

    if not video_metadata or not video_metadata.get("metadata", {}).get("path"):
        return VIDEO_PLACEHOLDER, "未找到同源视频。"

    metadata = video_metadata["metadata"]
    # 进度条现在使用视频绝对帧坐标，不再重复加 offset
    absolute_video_frame = int(frame_idx or 0)
    offset = video_metadata.get("offset", 0)
    
    safe_frame = max(0, min(absolute_video_frame, metadata["frame_count"] - 1))
    frame_src, error = encode_video_frame(metadata["path"], safe_frame)
    fps_text = f"{metadata.get('fps', 0):.2f}" if metadata.get("fps") else "未知"
    status = (
        f"视角: {video_metadata.get('camera', 'cam_1')} | 对齐偏移: {offset}\n"
        f"视频: {metadata['path']}\n"
        f"帧数: {metadata['frame_count']} | FPS: {fps_text} | 视频当前帧: {safe_frame}"
    )
    if error:
        status += f"\n{error}"
    return frame_src, status

# 修改保存动作：保存并自动跳转到下一个
@app.callback(
    [Output('save-result-msg', 'children'),
     Output('template-dropdown', 'value')],
    Input('save-label-btn', 'n_clicks'),
    [State('template-dropdown', 'value'), 
     State('dropdown-jump-type', 'value'),
     State('dropdown-stage', 'value'),
     State('radio-temporal-flag', 'value'),
     State('dropdown-action-units', 'value'),
     State('template-video-camera-dropdown', 'value')], # 💡 增加状态读取：当前视角
    prevent_initial_call=True
)
def save_custom_label(n_clicks, cid, jump, stage, temporal, units, camera):
    if not cid:
        # 如果是首次加载且没有选中值，默认选第一个
        db_init = load_db()
        return "目标字典未选定！", list(db_init.keys())[0] if db_init else dash.no_update
        
    if not jump or not stage:
        return "⚠️ 跳跃类型和阶段划分为必填项！", cid
        
    j_str = jump
    s_str = stage
    t_str = temporal if temporal else "S"
    
    # 💡 改进：保存时也进行自动排序
    if units:
        order = {'H': 0, 'T': 1, 'L': 2, 'R': 3, 'l': 4, 'r': 5}
        sorted_units = sorted(units, key=lambda x: (order.get(x[0], 99), int(x[1:]) if x[1:].isdigit() else 0))
        u_str = "[" + ",".join(sorted_units) + "]"
    else:
        u_str = "[]"
    
    new_label = f"{j_str}-{s_str}-{t_str}-{u_str}"
    
    db = load_db()
    if cid in db:
        db[cid]['label'] = new_label
        db[cid]['metadata'] = {
            'jump_type': jump,
            'stage': stage,
            'temporal_flag': temporal,
            'action_units': units if units else [],
            'camera': camera # 💡 记录视角，但这不会被合成到上面的 new_label 中
        }
        save_db(db)
        
        # 💡 极速标注改进：计算下一个 ID 以便通过控制 Output 实现跳转
        all_ids = sorted(db.keys(), key=lambda x: int(x))
        try:
            curr_idx = all_ids.index(cid)
            next_id = all_ids[curr_idx + 1] if curr_idx + 1 < len(all_ids) else cid
            msg = f"✅ 已保存 {cid}。自动跳转至 {next_id}"
            if next_id == cid: msg = "🎉 已全部标注完成！"
            return msg, next_id
        except:
            return f"已成功保存 {cid}", cid
            
    return "字典记录定位失败！", cid

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
        data = load_pose_sequence(file_path)
        frame_cnt = data.shape[0]
        new_slider = dcc.Slider(
            id='frame-slider', min=0, max=frame_cnt-1, step=1, value=0, 
            marks={0: '0', frame_cnt-1: str(frame_cnt-1)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
        return [new_slider]
    except Exception as e:
        return [html.Div(f"解析姿态文件遇到错误：{e}", style={'color': 'red'})]


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
        data = load_pose_sequence(file_path)
        raw_pose = data[frame_idx]
    except Exception as e:
         return dash.no_update, dash.no_update, f"抽取该帧错误: {e}"

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
