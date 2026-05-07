"""
动作模板标注与单帧比照台（Dash 单页面应用）。

功能 A：动作模板标注大厅
- 从 action_templates.json 中选择簇（Cluster）
- 预览该簇的标准模板骨架
- 编辑并保存标签回写到 JSON

功能 B：单帧动作鉴定比照台
- 选择测试 npy 文件
- 通过滑条选择帧
- 对齐 + 骨架归一化后，与模板库做最近邻距离比对
- 左右展示输入帧与匹配模板帧，并输出预测标签
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, dcc, html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from npy_loader import normalize_skeleton  # noqa: E402


H36M_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (0, 4),
    (0, 7),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (7, 8),
    (8, 9),
    (8, 11),
    (8, 14),
    (9, 10),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
]


def align_skeleton(frame: np.ndarray) -> np.ndarray:
    """
    与 action_space_explorer 保持一致：
    1) hip 中心化
    2) 骨盆向量旋转到 X 轴
    3) 骨架归一化
    """
    centered = np.asarray(frame, dtype=np.float64) - np.asarray(frame, dtype=np.float64)[0]
    v_hip = centered[4] - centered[1]
    v_hip_xy = np.array([v_hip[0], v_hip[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(v_hip_xy))
    if norm < 1e-8:
        return normalize_skeleton(centered)

    v_hip_norm = v_hip_xy / norm
    theta = float(np.arctan2(v_hip_norm[1], v_hip_norm[0]))
    c, s = np.cos(-theta), np.sin(-theta)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    aligned = centered @ rotation.T
    return normalize_skeleton(aligned)


def cluster_sort_key(cluster_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(cluster_id):06d}")
    except ValueError:
        return (1, cluster_id)


def load_template_db(template_path: Path) -> Dict[str, Dict[str, object]]:
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("action_templates.json 结构错误：顶层必须是对象（dict）")
    return payload


def build_template_matrix(template_db: Dict[str, Dict[str, object]]) -> tuple[np.ndarray, List[str]]:
    ordered_ids = sorted(template_db.keys(), key=cluster_sort_key)
    vectors: List[np.ndarray] = []
    valid_ids: List[str] = []
    for cluster_id in ordered_ids:
        skeleton = template_db.get(cluster_id, {}).get("skeleton")
        if skeleton is None:
            continue
        arr = np.asarray(skeleton, dtype=np.float64)
        if arr.shape != (17, 3):
            continue
        vectors.append(arr.reshape(-1))
        valid_ids.append(cluster_id)

    if not vectors:
        raise ValueError("模板库中没有有效 skeleton（要求 shape 为 17x3）")
    return np.vstack(vectors), valid_ids


def discover_npy_files(search_roots: List[Path]) -> List[str]:
    results: List[str] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for dir_path, _, file_names in os.walk(root):
            for name in file_names:
                if not name.endswith(".npy"):
                    continue
                abs_path = str((Path(dir_path) / name).resolve())
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                results.append(abs_path)
    results.sort()
    return results


def make_pose_figure(frame: np.ndarray, title: str, joint_color: str = "black", line_color: str = "red") -> go.Figure:
    joints = go.Scatter3d(
        x=frame[:, 0],
        y=frame[:, 1],
        z=frame[:, 2],
        mode="markers+text",
        marker=dict(size=4, color=joint_color),
        text=[str(i) for i in range(17)],
        textposition="top center",
        name="joints",
    )
    traces = [joints]
    for start_idx, end_idx in H36M_CONNECTIONS:
        traces.append(
            go.Scatter3d(
                x=[frame[start_idx, 0], frame[end_idx, 0], None],
                y=[frame[start_idx, 1], frame[end_idx, 1], None],
                z=[frame[start_idx, 2], frame[end_idx, 2], None],
                mode="lines",
                line=dict(color=line_color, width=4),
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def build_cluster_options(template_db: Dict[str, Dict[str, object]]) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for cluster_id in sorted(template_db.keys(), key=cluster_sort_key):
        label = str(template_db[cluster_id].get("label", "")).strip()
        show = f"Cluster {cluster_id}"
        if label:
            show += f" | {label}"
        options.append({"label": show, "value": cluster_id})
    return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动作模板标注与单帧比照台")
    parser.add_argument(
        "--template-file",
        default="output/action_templates.json",
        help="模板字典 JSON 路径（默认 output/action_templates.json）",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="测试 npy 搜索目录（可重复传入；默认 data/npy 与 data/npy_test）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8063, help="Dash port")
    parser.add_argument("--debug", action="store_true", help="Dash debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    template_path = Path(args.template_file)
    template_db = load_template_db(template_path)
    template_matrix, template_ids = build_template_matrix(template_db)

    search_roots = (
        [Path(p) for p in args.search_root]
        if args.search_root
        else [PROJECT_ROOT / "data" / "npy", PROJECT_ROOT / "data" / "npy_test"]
    )
    npy_files = discover_npy_files(search_roots)
    file_options = [{"label": p, "value": p} for p in npy_files]

    frame_cache: Dict[str, np.ndarray] = {}

    def _load_frames(path: str) -> np.ndarray:
        cached = frame_cache.get(path)
        if cached is not None:
            return cached
        arr = np.load(path)
        frame_cache[path] = arr
        if len(frame_cache) > 3:
            # 简单缓存上限，避免长期占用太多内存
            first_key = next(iter(frame_cache))
            frame_cache.pop(first_key, None)
        return arr

    app = dash.Dash(__name__)
    app.title = "Template Annotator"

    cluster_options = build_cluster_options(template_db)
    default_cluster = cluster_options[0]["value"] if cluster_options else None
    default_template = np.asarray(template_db[default_cluster]["skeleton"], dtype=np.float64) if default_cluster else np.zeros((17, 3))

    app.layout = html.Div(
        [
            html.H2("动作模板标注与单帧鉴定台"),
            dcc.Store(id="loaded-file-store", data={"path": "", "frame_count": 0}),
            dcc.Tabs(
                id="main-tabs",
                value="tab-annotate",
                children=[
                    dcc.Tab(
                        label="功能 A：动作模板标注大厅",
                        value="tab-annotate",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("选择模板簇"),
                                            dcc.Dropdown(
                                                id="cluster-dropdown",
                                                options=cluster_options,
                                                value=default_cluster,
                                                clearable=False,
                                            ),
                                            html.Br(),
                                            html.Label("动作标签"),
                                            dcc.Input(id="cluster-label-input", type="text", value="", style={"width": "100%"}),
                                            html.Br(),
                                            html.Hr(),
                                            html.Div(
                                                id="template-source-info",
                                                style={"whiteSpace": "pre-wrap", "fontSize": "13px", "color": "#555"},
                                            ),
                                            html.Br(),
                                            html.Br(),
                                            html.Button("保存标签到 JSON", id="save-label-btn", n_clicks=0),
                                            html.Div(id="save-label-status", style={"marginTop": "10px"}),
                                        ],
                                        style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "16px"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="template-pose-graph",
                                                figure=make_pose_figure(default_template, "模板骨架预览"),
                                                style={"height": "82vh"},
                                            )
                                        ],
                                        style={"width": "75%", "display": "inline-block", "verticalAlign": "top"},
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="功能 B：单帧动作鉴定比照台",
                        value="tab-compare",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("测试文件（下拉选择）"),
                                            dcc.Dropdown(
                                                id="test-file-dropdown",
                                                options=file_options,
                                                value=file_options[0]["value"] if file_options else None,
                                                clearable=True,
                                                placeholder="选择一个 .npy 文件",
                                            ),
                                            html.Br(),
                                            html.Label("或手动输入完整路径"),
                                            dcc.Input(
                                                id="test-file-input",
                                                type="text",
                                                value="",
                                                placeholder="D:/.../xx_Skater.npy",
                                                style={"width": "100%"},
                                            ),
                                            html.Br(),
                                            html.Br(),
                                            html.Button("加载测试文件", id="load-test-file-btn", n_clicks=0),
                                            html.Div(id="test-file-status", style={"marginTop": "10px"}),
                                        ],
                                        style={"padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "6px"},
                                    ),
                                    html.Br(),
                                    html.Label("选择测试帧"),
                                    dcc.Slider(
                                        id="frame-slider",
                                        min=0,
                                        max=0,
                                        step=1,
                                        value=0,
                                        marks={0: "0"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.H3(id="prediction-title", style={"marginTop": "16px"}),
                                    html.Div(
                                        [
                                            html.Div(
                                                [dcc.Graph(id="query-pose-graph", style={"height": "70vh"})],
                                                style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                                            ),
                                            html.Div(
                                                [dcc.Graph(id="matched-template-graph", style={"height": "70vh"})],
                                                style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                                            ),
                                        ]
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                ],
            ),
        ],
        style={"padding": "12px"},
    )

    @app.callback(
        Output("template-pose-graph", "figure"),
        Output("cluster-label-input", "value"),
        Output("template-source-info", "children"),
        Input("cluster-dropdown", "value"),
    )
    def update_template_panel(cluster_id: str):
        if not cluster_id or cluster_id not in template_db:
            empty = np.zeros((17, 3), dtype=np.float64)
            return make_pose_figure(empty, "模板骨架预览（无数据）"), "", "原始数据来源：无"
        entry = template_db[cluster_id]
        skeleton = np.asarray(entry.get("skeleton", np.zeros((17, 3))), dtype=np.float64)
        fig = make_pose_figure(skeleton, f"Cluster {cluster_id} 模板骨架")
        label = str(entry.get("label", "")).strip()
        source_file = str(entry.get("source_file", "N/A"))
        source_path = str(entry.get("source_path", "N/A"))
        frame_idx = entry.get("frame_idx", "N/A")
        source_info = (
            "原始数据来源\n"
            f"- source_file: {source_file}\n"
            f"- frame_idx: {frame_idx}\n"
            f"- source_path: {source_path}"
        )
        return fig, label, source_info

    @app.callback(
        Output("save-label-status", "children"),
        Output("cluster-dropdown", "options"),
        Input("save-label-btn", "n_clicks"),
        State("cluster-dropdown", "value"),
        State("cluster-label-input", "value"),
        prevent_initial_call=True,
    )
    def save_template_label(_: int, cluster_id: str, new_label: str):
        if not cluster_id or cluster_id not in template_db:
            return "保存失败：无效的 cluster。", build_cluster_options(template_db)
        template_db[cluster_id]["label"] = (new_label or "").strip()
        template_path.write_text(json.dumps(template_db, ensure_ascii=False, indent=4), encoding="utf-8")
        return f"已保存 Cluster {cluster_id} 标签：{template_db[cluster_id]['label']}", build_cluster_options(template_db)

    @app.callback(
        Output("loaded-file-store", "data"),
        Output("test-file-status", "children"),
        Output("frame-slider", "max"),
        Output("frame-slider", "marks"),
        Output("frame-slider", "value"),
        Input("load-test-file-btn", "n_clicks"),
        State("test-file-dropdown", "value"),
        State("test-file-input", "value"),
        prevent_initial_call=True,
    )
    def load_test_file(_: int, selected_file: str | None, typed_file: str):
        chosen = (typed_file or "").strip() or (selected_file or "")
        if not chosen:
            return {"path": "", "frame_count": 0}, "请先选择或输入测试文件路径。", 0, {0: "0"}, 0
        path = Path(chosen)
        if not path.exists() or path.suffix.lower() != ".npy":
            return {"path": "", "frame_count": 0}, f"文件不可用：{chosen}", 0, {0: "0"}, 0

        try:
            frames = _load_frames(str(path.resolve()))
        except Exception as exc:  # noqa: BLE001
            return {"path": "", "frame_count": 0}, f"加载失败：{exc}", 0, {0: "0"}, 0

        if frames.ndim != 3 or frames.shape[1:] != (17, 3):
            return {"path": "", "frame_count": 0}, f"数据形状非法：{frames.shape}，期望 (N,17,3)", 0, {0: "0"}, 0

        frame_count = int(frames.shape[0])
        max_idx = max(0, frame_count - 1)
        marks = {0: "0", max_idx: str(max_idx)} if max_idx > 0 else {0: "0"}
        status = f"已加载：{path}，共 {frame_count} 帧。"
        return {"path": str(path.resolve()), "frame_count": frame_count}, status, max_idx, marks, 0

    @app.callback(
        Output("prediction-title", "children"),
        Output("query-pose-graph", "figure"),
        Output("matched-template-graph", "figure"),
        Input("frame-slider", "value"),
        Input("loaded-file-store", "data"),
    )
    def compare_frame(frame_idx: int, loaded: Dict[str, object]):
        path = str((loaded or {}).get("path", ""))
        frame_count = int((loaded or {}).get("frame_count", 0))
        empty = make_pose_figure(np.zeros((17, 3), dtype=np.float64), "等待加载测试帧")
        if not path or frame_count <= 0:
            return "请先加载测试文件。", empty, empty

        frame_idx = int(np.clip(frame_idx or 0, 0, frame_count - 1))
        try:
            frames = _load_frames(path)
            query_raw = np.asarray(frames[frame_idx], dtype=np.float64)
            query_aligned = align_skeleton(query_raw)
        except Exception as exc:  # noqa: BLE001
            return f"测试帧处理失败：{exc}", empty, empty

        query_vec = query_aligned.reshape(1, -1)
        distances = np.linalg.norm(template_matrix - query_vec, axis=1)
        best_pos = int(np.argmin(distances))
        best_cluster = template_ids[best_pos]
        best_distance = float(distances[best_pos])
        best_entry = template_db.get(best_cluster, {})
        best_label = str(best_entry.get("label", "")).strip() or "未命名"
        template_pose = np.asarray(best_entry.get("skeleton", np.zeros((17, 3))), dtype=np.float64)

        title = f"该输入姿势被推断为：Cluster {best_cluster} 动作 —— 【{best_label}】"
        left_fig = make_pose_figure(query_aligned, f"输入测试帧（第 {frame_idx} 帧）", joint_color="black", line_color="royalblue")
        right_fig = make_pose_figure(template_pose, f"模板孪生骨架（Cluster {best_cluster}）", joint_color="black", line_color="crimson")
        right_fig.add_annotation(
            text=f"L2 距离：{best_distance:.4f}",
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.02,
            showarrow=False,
            font=dict(size=12, color="gray"),
        )
        return title, left_fig, right_fig

    print("=" * 72)
    print("Template Annotator 已启动")
    print(f"模板库: {template_path.resolve()}")
    print(f"模板数: {len(template_ids)}")
    print(f"测试文件候选数: {len(file_options)}")
    print(f"访问地址: http://{args.host}:{args.port}")
    print("=" * 72)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

