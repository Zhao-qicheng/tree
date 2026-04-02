"""
姿态模板审查与人工命名界面。

左侧：
- 簇内样本选择
- 当前样本骨架图

右侧：
- 模板选择与命名
- 模板摘要
- medoid 姿态
- 匿名规则
- 特征统计与差异特征
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from npy_loader import load_npy_file
from rule_classifier.pose_features import preprocess_frame

dash = None
Input = None
Output = None
State = None
dcc = None
html = None
go = None

H36M_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
    (12, 13), (14, 15), (15, 16),
]


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_assignments(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _make_pose_figure(frame: np.ndarray, title: str):
    joints = go.Scatter3d(
        x=frame[:, 0],
        y=frame[:, 1],
        z=frame[:, 2],
        mode="markers+text",
        marker=dict(size=3, color="black"),
        text=[str(i) for i in range(frame.shape[0])],
        textposition="top center",
        name="joints",
    )
    bones = []
    for start_idx, end_idx in H36M_CONNECTIONS:
        bones.append(
            go.Scatter3d(
                x=[frame[start_idx, 0], frame[end_idx, 0], None],
                y=[frame[start_idx, 1], frame[end_idx, 1], None],
                z=[frame[start_idx, 2], frame[end_idx, 2], None],
                mode="lines",
                line=dict(color="red", width=4),
                showlegend=False,
            )
        )
    fig = go.Figure(data=[joints] + bones)
    fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40), scene=dict(aspectmode="cube"))
    return fig


def _feature_table(features: Dict[str, float]):
    rows = [html.Tr([html.Th("Feature"), html.Th("Mean"), html.Th("Std")])]
    for feature_name in sorted(features["mean"].keys()):
        rows.append(
            html.Tr(
                [
                    html.Td(feature_name),
                    html.Td(f"{features['mean'][feature_name]:.4f}"),
                    html.Td(f"{features['std'][feature_name]:.4f}"),
                ]
            )
        )
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def _top_feature_table(items: List[Dict[str, float]]):
    rows = [html.Tr([html.Th("Feature"), html.Th("Effect Size"), html.Th("Delta")])]
    for item in items[:8]:
        rows.append(
            html.Tr(
                [
                    html.Td(item.get("feature", "")),
                    html.Td(f"{float(item.get('effect_size', 0.0)):.4f}"),
                    html.Td(f"{float(item.get('delta', 0.0)):.4f}"),
                ]
            )
        )
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def _load_member_frame(member: Dict[str, str]) -> np.ndarray:
    source_file = member["source_file"]
    frame_index = int(member["frame_index"])
    frames = load_npy_file(source_file)
    return preprocess_frame(frames[frame_index], align=True, normalize=True)


def main() -> None:
    global dash, Input, Output, State, dcc, html, go

    parser = argparse.ArgumentParser(description="姿态模板审查界面")
    parser.add_argument("--output-dir", default="output/pose_templates", help="模板输出目录")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8060, help="Dash port")
    parser.add_argument("--debug", action="store_true", help="Dash debug")
    args = parser.parse_args()

    try:
        import dash as dash_lib
        from dash import Input as DashInput, Output as DashOutput, State as DashState, dcc as dash_dcc, html as dash_html
        import plotly.graph_objects as go_lib
    except ImportError as exc:
        raise ImportError("缺少可视化依赖。请先安装：pip install dash plotly") from exc

    dash = dash_lib
    Input = DashInput
    Output = DashOutput
    State = DashState
    dcc = dash_dcc
    html = dash_html
    go = go_lib

    output_dir = Path(args.output_dir)
    templates_data = _read_json(str(output_dir / "pose_templates.json"))
    rules_data = _read_json(str(output_dir / "anonymous_rules.json"))
    quality_data = _read_json(str(output_dir / "template_quality_report.json"))
    assignments = _read_assignments(str(output_dir / "cluster_assignments.csv"))
    cluster_names_path = output_dir / "cluster_names.json"
    cluster_names = _read_json(str(cluster_names_path)) if cluster_names_path.exists() else {}
    metadata_summary = templates_data.get("metadata_summary", {})

    templates = {item["template_id"]: item for item in templates_data["templates"]}
    rules = {item["template_id"]: item for item in rules_data.get("rules", [])}

    members_by_template: Dict[str, List[Dict[str, str]]] = {}
    for row in assignments:
        template_id = row.get("template_id", "")
        if template_id:
            members_by_template.setdefault(template_id, []).append(row)

    default_template_id = next(iter(templates))

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("姿态模板审查与命名"),
            html.Div(
                [
                    html.H4("簇内样本"),
                    dcc.Dropdown(id="member-dropdown", clearable=False),
                    dcc.Graph(id="member-graph", style={"height": "88vh", "marginTop": "12px"}),
                ],
                style={"width": "32%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "16px"},
            ),
            html.Div(
                [
                    html.Label("选择模板"),
                    dcc.Dropdown(
                        id="template-dropdown",
                        options=[
                            {
                                "label": f"{template_id} | {cluster_names.get(template_id, '')}".strip(),
                                "value": template_id,
                            }
                            for template_id in templates.keys()
                        ],
                        value=default_template_id,
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label("人工命名"),
                    dcc.Input(id="cluster-name-input", type="text", value=cluster_names.get(default_template_id, ""), debounce=True),
                    html.Button("保存命名", id="save-name-btn", n_clicks=0, style={"marginLeft": "8px"}),
                    html.Div(id="save-status", style={"marginTop": "8px"}),
                    html.Hr(),
                    html.Div(id="template-summary"),
                    dcc.Graph(id="medoid-graph", style={"height": "42vh", "marginTop": "12px"}),
                    html.Hr(),
                    html.H4("匿名规则"),
                    html.Pre(id="rule-text", style={"whiteSpace": "pre-wrap"}),
                    html.H4("模板特征统计"),
                    html.Div(id="feature-stats"),
                    html.H4("主要差异特征"),
                    html.Div(id="top-features"),
                ],
                style={"width": "68%", "display": "inline-block", "verticalAlign": "top"},
            ),
        ],
        style={"padding": "12px 16px"},
    )

    @app.callback(
        Output("member-dropdown", "options"),
        Output("member-dropdown", "value"),
        Output("cluster-name-input", "value"),
        Input("template-dropdown", "value"),
    )
    def update_member_dropdown(template_id: str):
        members = members_by_template.get(template_id, [])
        options = [
            {
                "label": f"{row['file_name']} | frame={row['frame_index']} | {row['skater']} | {row['jump_type']}",
                "value": f"{row['source_file']}|{row['frame_index']}",
            }
            for row in members[:200]
        ]
        value = options[0]["value"] if options else None
        return options, value, cluster_names.get(template_id, "")

    @app.callback(
        Output("template-summary", "children"),
        Output("rule-text", "children"),
        Output("feature-stats", "children"),
        Output("top-features", "children"),
        Output("medoid-graph", "figure"),
        Output("member-graph", "figure"),
        Input("template-dropdown", "value"),
        Input("member-dropdown", "value"),
    )
    def update_template_view(template_id: str, member_value: str):
        template = templates[template_id]
        medoid_frame = np.array(template["medoid_pose"], dtype=np.float64)
        medoid_figure = _make_pose_figure(medoid_frame, f"{template_id} medoid")

        if member_value:
            source_file, frame_index = member_value.rsplit("|", 1)
            member_frame = _load_member_frame({"source_file": source_file, "frame_index": frame_index})
            member_figure = _make_pose_figure(member_frame, f"{Path(source_file).name} | frame={frame_index}")
        else:
            member_figure = _make_pose_figure(medoid_frame, f"{template_id} sample")

        summary_lines = [
            html.Div(f"模板ID: {template_id}"),
            html.Div(f"人工名称: {cluster_names.get(template_id, '未命名')}"),
            html.Div(f"特征源: {metadata_summary.get('feature_source', 'unknown')}"),
            html.Div(f"坐标模式: {metadata_summary.get('coordinate_mode', 'n/a')}"),
            html.Div(f"八叉树模式: {metadata_summary.get('octree_node_mode', 'n/a')}"),
            html.Div(f"使用 Z-score: {metadata_summary.get('use_zscore', False)}"),
            html.Div(f"使用 PCA: {metadata_summary.get('use_pca', False)}"),
            html.Div(f"簇大小: {template['cluster_size']}"),
            html.Div(f"簇半径: {float(template['cluster_radius']):.4f}"),
            html.Div(f"质量标记: {', '.join(template.get('quality_flags', [])) or '无'}"),
        ]
        if metadata_summary.get("feature_source") != "handcrafted":
            summary_lines.append(html.Div("提示: 当前规则为实验性结果，可解释性较弱。"))
        split_ids = {item["template_id"] for item in quality_data.get("split_candidates", [])}
        merge_pairs = [
            f"{item['template_a']} <-> {item['template_b']} ({float(item['distance']):.4f})"
            for item in quality_data.get("merge_candidates", [])
            if template_id in {item["template_a"], item["template_b"]}
        ]
        if template_id in split_ids:
            summary_lines.append(html.Div("建议: 当前模板需要进一步细分"))
        if merge_pairs:
            summary_lines.append(html.Div("可疑近邻模板: " + "; ".join(merge_pairs)))

        rule = rules.get(template_id, {})
        rule_text = rule.get("rule_text", f"THEN {template_id}")
        feature_stats = _feature_table({"mean": template["mean_feature"], "std": template["std_feature"]})
        top_features = _top_feature_table(rule.get("top_features", []))
        return summary_lines, rule_text, feature_stats, top_features, medoid_figure, member_figure

    @app.callback(
        Output("save-status", "children"),
        Input("save-name-btn", "n_clicks"),
        State("template-dropdown", "value"),
        State("cluster-name-input", "value"),
        prevent_initial_call=True,
    )
    def save_cluster_name(_: int, template_id: str, cluster_name: str):
        cluster_names[template_id] = (cluster_name or "").strip()
        cluster_names_path.write_text(json.dumps(cluster_names, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"已保存 {template_id} -> {cluster_names[template_id] or '空名称'}"

    print("=" * 72)
    print("姿态模板审查界面已启动")
    print(f"输出目录: {output_dir}")
    print(f"访问地址: http://{args.host}:{args.port}")
    print("=" * 72)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

