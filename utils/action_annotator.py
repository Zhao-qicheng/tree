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
    {"label": "7: 落冰后滑行", "value": "7"},
    {"label": "8: 滑出", "value": "8"}
]

TEMPORAL_FLAGS = [
    {"label": "单帧独立判定 (F)", "value": "F"},
    {"label": "需要时序相邻帧 (W)", "value": "W"}
]

ACTION_UNITS = [
    {"label": "H0: 头部直视/无明显动作", "value": "H0"},
    {"label": "H1: 头部左旋转", "value": "H1"},
    {"label": "H2: 头部右旋转", "value": "H2"},
    {"label": "H3: 头上仰", "value": "H3"},
    {"label": "H4: 头下低", "value": "H4"},
    
    {"label": "B0: 躯干直立/无明显动作", "value": "B0"},
    {"label": "B1: 躯干左旋转", "value": "B1"},
    {"label": "B2: 躯干右旋转", "value": "B2"},
    {"label": "B3: 躯干前倾", "value": "B3"},
    {"label": "B4: 躯干后仰", "value": "B4"},
    {"label": "B5: 躯干左倾斜", "value": "B5"},
    {"label": "B6: 躯干右倾斜", "value": "B6"},
    
    {"label": "LS0: 左上臂下垂/无明显动作", "value": "LS0"},
    {"label": "LS1: 左上臂前屈", "value": "LS1"},
    {"label": "LS2: 左上臂外展", "value": "LS2"},
    {"label": "LS3: 左上臂后伸", "value": "LS3"},
    {"label": "LS4: 左上臂内收", "value": "LS4"},
    {"label": "LE0: 左肘微弯/无明显动作", "value": "LE0"},
    {"label": "LE1: 左肘弯曲", "value": "LE1"},
    {"label": "LE2: 左肘伸直", "value": "LE2"},
    
    {"label": "RS0: 右上臂下垂/无明显动作", "value": "RS0"},
    {"label": "RS1: 右上臂前屈", "value": "RS1"},
    {"label": "RS2: 右上臂外展", "value": "RS2"},
    {"label": "RS3: 右上臂后伸", "value": "RS3"},
    {"label": "RS4: 右上臂内收", "value": "RS4"},
    {"label": "RE0: 右肘微弯/无明显动作", "value": "RE0"},
    {"label": "RE1: 右肘弯曲", "value": "RE1"},
    {"label": "RE2: 右肘伸直", "value": "RE2"},
    
    {"label": "LH0: 左髋直立/无明显动作", "value": "LH0"},
    {"label": "LH1: 左髋前屈", "value": "LH1"},
    {"label": "LH2: 左髋外展", "value": "LH2"},
    {"label": "LH3: 左髋后伸", "value": "LH3"},
    {"label": "LH4: 左髋内收", "value": "LH4"},
    {"label": "LK0: 左膝微弯/无明显动作", "value": "LK0"},
    {"label": "LK1: 左膝弯曲", "value": "LK1"},
    {"label": "LK2: 左膝伸直", "value": "LK2"},
    {"label": "LF1: 左脚点冰（脚尖下压）", "value": "LF1"},
    {"label": "LF2: 左脚平滑", "value": "LF2"},
    
    {"label": "RH0: 右髋直立/无明显动作", "value": "RH0"},
    {"label": "RH1: 右髋前屈", "value": "RH1"},
    {"label": "RH2: 右髋外展", "value": "RH2"},
    {"label": "RH3: 右髋后伸", "value": "RH3"},
    {"label": "RH4: 右髋内收", "value": "RH4"},
    {"label": "RK0: 右膝微弯/无明显动作", "value": "RK0"},
    {"label": "RK1: 右膝弯曲", "value": "RK1"},
    {"label": "RK2: 右膝伸直", "value": "RK2"},
    {"label": "RF1: 点冰（脚尖下压）", "value": "RF1"},
    {"label": "RF2: 右脚滑行", "value": "RF2"}
]
ACTION_UNIT_VALUES = {item["value"] for item in ACTION_UNITS}
ACTION_UNIT_LABELS = {item["value"]: item["label"] for item in ACTION_UNITS}

DEFAULT_TEMPORAL_FLAG = "F"
TEMPORAL_FLAG_ALIASES = {"S": "F", "T": "W"}
ACTION_UNIT_PREFIX_ORDER = ["H", "B", "LS", "LE", "RS", "RE", "LH", "LK", "LF", "RH", "RK", "RF"]
ACTION_UNIT_ALIASES = {
    "H5": "H0",
    "T1": "B1", "T2": "B2", "T3": "B3", "T4": "B4", "T5": "B5", "T6": "B6", "T7": "B0",
    "L1": "LS1", "L2": "LS2", "L3": "LS3", "L4": "LS4", "L7": "LS0",
    "R1": "RS1", "R2": "RS2", "R3": "RS3", "R4": "RS4", "R7": "RS0",
    "l1": "LH1", "l2": "LH2", "l3": "LH3", "l4": "LH4", "l5": "LK1", "l6": "LF1", "l8": "LH0",
    "r1": "RH1", "r2": "RH2", "r3": "RH3", "r4": "RH4", "r5": "RK1", "r6": "RF1", "r8": "RH0",
}


def normalize_temporal_flag(value):
    if not value:
        return DEFAULT_TEMPORAL_FLAG
    return TEMPORAL_FLAG_ALIASES.get(value, value)


def action_unit_sort_key(unit):
    for prefix_index, prefix in enumerate(ACTION_UNIT_PREFIX_ORDER):
        if unit.startswith(prefix):
            suffix = unit[len(prefix):]
            number = int(suffix) if suffix.isdigit() else 0
            return prefix_index, number, unit
    return len(ACTION_UNIT_PREFIX_ORDER), 0, unit


def normalize_action_units(units):
    if not units:
        return []
    normalized = []
    seen = set()
    for unit in units:
        mapped = ACTION_UNIT_ALIASES.get(unit, unit)
        if mapped not in seen:
            normalized.append(mapped)
            seen.add(mapped)
    return sorted(normalized, key=action_unit_sort_key)


def cluster_sort_key(cluster_id):
    try:
        return (0, int(cluster_id))
    except (TypeError, ValueError):
        return (1, str(cluster_id))


def extract_action_units_from_label(label):
    if not label:
        return []

    label_text = str(label)
    start = label_text.find("[")
    end = label_text.find("]", start + 1)
    if start < 0 or end < 0:
        return []

    raw_units = [unit.strip() for unit in label_text[start + 1:end].split(",") if unit.strip()]
    return [unit for unit in normalize_action_units(raw_units) if unit in ACTION_UNIT_VALUES]


def get_template_action_units(template_data):
    metadata = template_data.get("metadata", {}) or {}
    units = metadata.get("action_units") or extract_action_units_from_label(template_data.get("label", ""))
    return [unit for unit in normalize_action_units(units) if unit in ACTION_UNIT_VALUES]


def search_templates_by_action_units(query_units, match_mode="all"):
    db = load_db()
    normalized_query = [unit for unit in normalize_action_units(query_units or []) if unit in ACTION_UNIT_VALUES]
    query_set = set(normalized_query)
    matches = []

    for cluster_id, template_data in db.items():
        template_units = get_template_action_units(template_data)
        if not template_units:
            continue

        template_set = set(template_units)
        if not query_set:
            is_match = True
        elif match_mode == "any":
            is_match = bool(query_set & template_set)
        elif match_mode == "exact":
            is_match = query_set == template_set
        else:
            is_match = query_set.issubset(template_set)

        if not is_match:
            continue

        overlap_count = len(query_set & template_set) if query_set else len(template_set)
        matches.append({
            "cluster_id": str(cluster_id),
            "data": template_data,
            "units": template_units,
            "overlap_count": overlap_count,
        })

    matches.sort(key=lambda item: (-item["overlap_count"], len(item["units"]), cluster_sort_key(item["cluster_id"])))
    return matches


def build_unit_search_option(match):
    cluster_id = match["cluster_id"]
    data = match["data"]
    units_text = ",".join(match["units"])
    source_file = data.get("source_file", "未知")
    frame_idx = data.get("frame_idx", "?")
    label = data.get("label", "未标注")
    return {
        "label": f"簇 {cluster_id} | 第 {frame_idx} 帧 | {source_file} | {label} | [{units_text}]",
        "value": cluster_id,
    }
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


def _safe_angle_degrees(a, b, c):
    """计算 a-b-c 三点在 b 点处的夹角。"""
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _classify_flexion(angle, bent_code, straight_code, neutral_code=None, notes=None, label="关节"):
    if angle is None:
        if notes is not None:
            notes.append(f"{label}角度不可用")
        return neutral_code or straight_code
    if angle < 130:
        return bent_code
    if angle < 160:
        return neutral_code or straight_code
    return straight_code


def _classify_upper_arm(pose, shoulder_idx, elbow_idx, side_code, notes, label):
    shoulder = pose[shoulder_idx]
    elbow = pose[elbow_idx]
    chest = pose[8]
    vec = elbow - shoulder
    length = np.linalg.norm(vec)
    if length < 1e-6:
        notes.append(f"{label}上臂向量不可用")
        return f"{side_code}0"

    x, y, z = vec / length
    side_sign = np.sign(shoulder[0] - chest[0])
    if side_sign == 0:
        side_sign = 1 if side_code == "LS" else -1

    horizontal = max(abs(x), abs(y))
    if abs(x) >= 0.30 and abs(x) >= abs(y):
        return f"{side_code}2" if np.sign(x) == side_sign else f"{side_code}4"
    # 当前标注页骨架对齐后，+X 约为身体左侧，-Y 约为身体前方。
    if y <= -0.38:
        return f"{side_code}1"
    if y >= 0.38:
        return f"{side_code}3"
    if z < -0.70 and horizontal < 0.45:
        return f"{side_code}0"

    notes.append(f"{label}上臂方向不明显，默认无明显动作")
    return f"{side_code}0"


def _classify_hip_motion(pose, hip_idx, knee_idx, side_code, notes, label):
    hip = pose[hip_idx]
    knee = pose[knee_idx]
    root = pose[0]
    vec = knee - hip
    length = np.linalg.norm(vec)
    if length < 1e-6:
        notes.append(f"{label}髋/大腿方向不可用")
        return f"{side_code}0"

    x, y, z = vec / length
    side_sign = np.sign(hip[0] - root[0])
    if side_sign == 0:
        side_sign = 1 if side_code == "LH" else -1

    if z < -0.70 and max(abs(x), abs(y)) < 0.48:
        return f"{side_code}0"
    if abs(y) >= abs(x) and abs(y) >= 0.35:
        return f"{side_code}1" if y < 0 else f"{side_code}3"
    if abs(x) >= 0.35:
        return f"{side_code}2" if np.sign(x) == side_sign else f"{side_code}4"

    notes.append(f"{label}髋/大腿方向不明显，默认无明显动作")
    return f"{side_code}0"


def _classify_torso(pose, notes):
    hip = pose[0]
    chest = pose[8]
    vec = chest - hip
    length = np.linalg.norm(vec)
    if length < 1e-6:
        notes.append("躯干方向不可用，默认 B0")
        return "B0"

    x, y, z = vec / length
    horizontal = max(abs(x), abs(y))
    if horizontal < 0.30 or abs(z) >= 0.88:
        return "B0"
    if abs(y) >= abs(x):
        return "B3" if y < 0 else "B4"
    return "B6" if x > 0 else "B5"


def _classify_head(pose, notes):
    neck = pose[9]
    head = pose[10]
    vec = head - neck
    length = np.linalg.norm(vec)
    if length < 1e-6:
        notes.append("头部方向不可用，默认 H0")
        return "H0"

    _, y, z = vec / length
    if z < 0.45:
        return "H4"
    if z > 0.80 and abs(y) > 0.35:
        return "H3"
    notes.append("头部姿态采用保守默认 H0")
    return "H0"


def _classify_foot_markers(foot_points, prefix, toe_name, heel_name, notes):
    if not foot_points:
        return None
    toe = foot_points.get(toe_name)
    heel = foot_points.get(heel_name)
    if toe is None or heel is None:
        notes.append(f"{prefix}脚部 marker 不完整，未推荐脚部动作")
        return None

    toe = np.asarray(toe, dtype=float)
    heel = np.asarray(heel, dtype=float)
    vertical_delta = toe[2] - heel[2]
    if vertical_delta < -35:
        return f"{prefix}1"
    if abs(vertical_delta) <= 25:
        return f"{prefix}2"
    notes.append(f"{prefix}脚尖/脚跟高度差不稳定，未推荐脚部动作")
    return None


def auto_recommend_action_units(skeleton, foot_points=None):
    pose = np.asarray(skeleton, dtype=float)
    notes = []
    if pose.shape != (17, 3):
        return {
            "action_units": [],
            "temporal_flag": DEFAULT_TEMPORAL_FLAG,
            "confidence": "低",
            "notes": [f"骨架形状应为 (17, 3)，实际为 {pose.shape}"],
        }

    units = [
        _classify_head(pose, notes),
        _classify_torso(pose, notes),
        _classify_upper_arm(pose, 11, 12, "LS", notes, "左"),
        _classify_flexion(_safe_angle_degrees(pose[11], pose[12], pose[13]), "LE1", "LE2", "LE0", notes, "左肘"),
        _classify_upper_arm(pose, 14, 15, "RS", notes, "右"),
        _classify_flexion(_safe_angle_degrees(pose[14], pose[15], pose[16]), "RE1", "RE2", "RE0", notes, "右肘"),
        _classify_hip_motion(pose, 4, 5, "LH", notes, "左"),
        _classify_flexion(_safe_angle_degrees(pose[4], pose[5], pose[6]), "LK1", "LK2", "LK0", notes, "左膝"),
        _classify_hip_motion(pose, 1, 2, "RH", notes, "右"),
        _classify_flexion(_safe_angle_degrees(pose[1], pose[2], pose[3]), "RK1", "RK2", "RK0", notes, "右膝"),
    ]

    left_foot = _classify_foot_markers(foot_points, "LF", "L_Toe", "LHEL", notes)
    right_foot = _classify_foot_markers(foot_points, "RF", "R_Toe", "RHEL", notes)
    if left_foot:
        units.append(left_foot)
    if right_foot:
        units.append(right_foot)

    units = normalize_action_units([unit for unit in units if unit in ACTION_UNIT_VALUES])
    if len(notes) <= 2:
        confidence = "高"
    elif len(notes) <= 5:
        confidence = "中"
    else:
        confidence = "低"

    return {
        "action_units": units,
        "temporal_flag": DEFAULT_TEMPORAL_FLAG,
        "confidence": confidence,
        "notes": notes,
    }


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
                    dcc.RadioItems(id='radio-temporal-flag', options=TEMPORAL_FLAGS, value=DEFAULT_TEMPORAL_FLAG, inline=True, style={'marginBottom': '10px'}),
                    
                    html.Label("动作单元 (多选)", style={'fontWeight': 'bold', 'marginTop': '10px', 'display': 'block'}),
                    dcc.Dropdown(id='dropdown-action-units', options=ACTION_UNITS, multi=True, placeholder="可选择多项组合...", closeOnSelect=False),
                    html.Button("自动推荐动作单元", id='auto-recommend-btn', n_clicks=0, style={'marginTop': '10px', 'padding': '8px 14px', 'backgroundColor': '#1971c2', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '5px', 'fontWeight': 'bold'}),
                    html.Div(id='auto-recommend-msg', style={'marginTop': '8px', 'whiteSpace': 'pre-wrap', 'color': '#495057', 'fontSize': '13px'}),
                    
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
        ]),

        # ====================== [板块 3] : 动作单元模板检索 ======================
        dcc.Tab(label='🔎 动作单元模板检索', children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("搜索动作单元", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            id='unit-search-dropdown',
                            options=ACTION_UNITS,
                            multi=True,
                            placeholder="选择一个或多个动作单元，例如 H0、B3、LK1...",
                            closeOnSelect=False,
                        )
                    ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        html.Label("匹配模式", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '6px'}),
                        dcc.RadioItems(
                            id='unit-search-mode',
                            options=[
                                {'label': '包含全部', 'value': 'all'},
                                {'label': '包含任一', 'value': 'any'},
                                {'label': '完全一致', 'value': 'exact'},
                            ],
                            value='all',
                            inline=True,
                            inputStyle={'marginRight': '4px', 'marginLeft': '10px'}
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
                ], style={'padding': '18px 20px', 'backgroundColor': '#f8f9fa', 'borderBottom': '1px solid #dee2e6'}),

                html.Div(id='unit-search-count', style={'padding': '10px 20px', 'fontWeight': 'bold', 'color': '#495057'}),

                html.Div([
                    html.Div([
                        html.H3("返回的模板帧", style={'marginTop': 0}),
                        dcc.RadioItems(
                            id='unit-search-result-list',
                            options=[],
                            value=None,
                            labelStyle={
                                'display': 'block',
                                'padding': '10px 12px',
                                'marginBottom': '8px',
                                'border': '1px solid #dee2e6',
                                'borderRadius': '6px',
                                'backgroundColor': 'white',
                                'cursor': 'pointer',
                                'lineHeight': '1.45'
                            },
                            inputStyle={'marginRight': '8px'}
                        )
                    ], style={
                        'width': '36%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'height': '68vh',
                        'overflowY': 'auto',
                        'padding': '20px',
                        'boxSizing': 'border-box',
                        'backgroundColor': '#f1f3f5'
                    }),

                    html.Div([
                        html.H3("选中模板骨架", style={'textAlign': 'center', 'margin': '0 0 8px 0'}),
                        dcc.Graph(id='unit-search-pose-graph', style={'height': '55vh', 'width': '100%'}),
                        html.Div(id='unit-search-template-info', style={
                            'whiteSpace': 'pre-wrap',
                            'backgroundColor': '#f8f9fa',
                            'padding': '14px',
                            'borderRadius': '6px',
                            'border': '1px solid #dee2e6',
                            'color': '#343a40'
                        })
                    ], style={
                        'width': '64%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'padding': '20px',
                        'boxSizing': 'border-box'
                    })
                ])
            ])
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
               for cid in sorted(db.keys(), key=cluster_sort_key)]
    return options


@app.callback(
    [Output('unit-search-result-list', 'options'),
     Output('unit-search-result-list', 'value'),
     Output('unit-search-count', 'children')],
    [Input('unit-search-dropdown', 'value'),
     Input('unit-search-mode', 'value')]
)
def update_unit_search_results(query_units, match_mode):
    db = load_db()
    if not db:
        return [], None, "模板库为空，请先保存带动作单元标签的模板帧。"

    matches = search_templates_by_action_units(query_units, match_mode)
    if not matches:
        return [], None, "未找到匹配模板。可以减少动作单元，或切换为“包含任一”。"

    options = [build_unit_search_option(match) for match in matches]
    selected_value = options[0]["value"]
    query_text = "、".join(normalize_action_units(query_units or [])) or "全部已标注动作单元"
    mode_text = {"all": "包含全部", "any": "包含任一", "exact": "完全一致"}.get(match_mode, "包含全部")
    return options, selected_value, f"搜索：{query_text} | 模式：{mode_text} | 返回 {len(options)} 个模板帧"


@app.callback(
    [Output('unit-search-pose-graph', 'figure'),
     Output('unit-search-template-info', 'children')],
    Input('unit-search-result-list', 'value')
)
def render_unit_search_template(cluster_id):
    if not cluster_id:
        return create_pose_figure(None, title="请选择左侧模板帧"), "请选择左侧返回的模板帧。"

    db = load_db()
    data = db.get(str(cluster_id))
    if not data:
        return create_pose_figure(None, title="模板不存在"), "模板库中未找到该记录。"

    try:
        skeleton = np.asarray(data.get('skeleton'), dtype=float)
    except Exception as exc:
        return create_pose_figure(None, title="骨架读取失败"), f"骨架数据读取失败：{exc}"

    source_path = data.get('source_path')
    source_file = data.get('source_file', '未知')
    frame_idx = int(data.get('frame_idx', 0) or 0)
    skip_video = is_skeleton_source(data)
    foot_points = {} if skip_video else load_foot_points(source_path, frame_idx, skeleton)
    fig = create_pose_figure(
        skeleton,
        title=f"动作单元检索结果 - 簇 {cluster_id}",
        color='#1971c2',
        extra_points=foot_points
    )

    units = get_template_action_units(data)
    unit_lines = []
    for unit in units:
        unit_lines.append(ACTION_UNIT_LABELS.get(unit, unit))

    info_text = f"模板标识：{data.get('label', '未标注')}\n"
    info_text += f"簇 ID：{cluster_id}\n"
    info_text += f"来源文件：{source_file}\n"
    info_text += f"所在原帧：第 {frame_idx} 帧\n"
    info_text += f"动作单元：{', '.join(units) if units else '无'}"
    if unit_lines:
        info_text += "\n动作单元说明：\n" + "\n".join(unit_lines)
    if skip_video:
        info_text += "\n视频：无（Skeleton NPZ）"
    elif foot_points:
        info_text += f"\n脚部 marker：已加载 {len(foot_points)} 个"

    return fig, info_text


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
    t_str = normalize_temporal_flag(temporal)
    
    # 按动作号码文件中的身体部位顺序排序，保证同一组动作编码稳定。
    if units:
        sorted_units = normalize_action_units(units)
        u_str = "[" + ",".join(sorted_units) + "]"
    else:
        u_str = "[]"
    
    preview_text = f"{j_str}-{s_str}-{t_str}-{u_str}"
    return f"🏷️ 预览编码格式: {preview_text}"


@app.callback(
    [Output('radio-temporal-flag', 'value', allow_duplicate=True),
     Output('dropdown-action-units', 'value', allow_duplicate=True),
     Output('auto-recommend-msg', 'children')],
    Input('auto-recommend-btn', 'n_clicks'),
    State('template-dropdown', 'value'),
    prevent_initial_call=True
)
def apply_auto_recommendation(n_clicks, cluster_id):
    if not cluster_id:
        return dash.no_update, dash.no_update, "请先选择一个动作集群。"

    db = load_db()
    data = db.get(cluster_id)
    if not data:
        return dash.no_update, dash.no_update, "未找到当前动作集群，无法推荐。"

    try:
        skeleton = np.asarray(data.get('skeleton'), dtype=float)
    except Exception as exc:
        return dash.no_update, dash.no_update, f"骨架数据读取失败：{exc}"

    source_path = data.get('source_path')
    frame_idx = int(data.get('frame_idx', 0) or 0)
    foot_points = {} if is_skeleton_source(data) else load_foot_points(source_path, frame_idx, skeleton)
    recommendation = auto_recommend_action_units(skeleton, foot_points=foot_points)
    units = recommendation["action_units"]
    temporal = recommendation["temporal_flag"]
    confidence = recommendation["confidence"]
    notes = recommendation["notes"]

    if units:
        unit_text = ",".join(units)
        msg = f"推荐完成：动作单元 {len(units)} 项，置信度：{confidence}。\n推荐：[{unit_text}]\n阶段建议人工确认。"
    else:
        msg = f"未生成有效动作单元，置信度：{confidence}。请人工标注。"
    if notes:
        msg += "\n注意：" + "；".join(notes[:4])

    return temporal, units, msg


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
            dash.no_update, "无展示数据。", None, None, DEFAULT_TEMPORAL_FLAG, [],
            discover_camera_options(), "cam_1", {}, "无展示数据。", VIDEO_PLACEHOLDER,
            controls["frame_min"], controls["frame_max"], controls["frame_value"], controls["frame_marks"], controls["frame_disabled"],
        )
    
    db = load_db()
    if cluster_id not in db:
        controls = build_video_control_props(None, 0)
        return (
            dash.no_update, "分类字典已损坏或缺失", None, None, DEFAULT_TEMPORAL_FLAG, [],
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
    t_val = normalize_temporal_flag(metadata.get('temporal_flag', DEFAULT_TEMPORAL_FLAG))
    u_val = normalize_action_units(metadata.get('action_units', []))

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
    t_str = normalize_temporal_flag(temporal)
    
    # 保存时也进行自动排序，和实时预览保持一致。
    if units:
        sorted_units = normalize_action_units(units)
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
            'temporal_flag': t_str,
            'action_units': sorted_units if units else [],
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
