"""
列出已打标签模板（来自 action_templates.json）。

默认行为：
- 读取 output/action_templates.json
- 筛选 label 非空项
- 可选忽略默认数字标签（label == cluster_id）
- 打印到终端
- 可选导出 CSV
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="列出 action_templates.json 中已打标签模板")
    parser.add_argument(
        "--template-file",
        default="output/action_templates.json",
        help="模板字典文件路径（默认 output/action_templates.json）",
    )
    parser.add_argument(
        "--include-default-numeric-label",
        action="store_true",
        help="包含默认数字标签（例如 cluster_id=12 且 label=12）",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="可选：导出 CSV 文件路径（例如 output/labeled_templates.csv）",
    )
    return parser.parse_args()


def _cluster_sort_key(cluster_id: str) -> tuple[int, int | str]:
    return (0, int(cluster_id)) if cluster_id.isdigit() else (1, cluster_id)


def load_template_db(template_path: Path) -> Dict[str, Dict[str, object]]:
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("action_templates.json 结构错误：顶层必须是对象(dict)")
    return payload


def collect_labeled_rows(
    template_db: Dict[str, Dict[str, object]],
    *,
    include_default_numeric_label: bool = False,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cluster_id, entry in template_db.items():
        label = str(entry.get("label", "")).strip()
        if not label:
            continue
        if not include_default_numeric_label and label == str(cluster_id):
            continue

        rows.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "source_file": str(entry.get("source_file", "")),
                "frame_idx": entry.get("frame_idx", ""),
                "source_path": str(entry.get("source_path", "")),
            }
        )
    rows.sort(key=lambda row: _cluster_sort_key(str(row["cluster_id"])))
    return rows


def print_rows(rows: List[Dict[str, object]]) -> None:
    print(f"已标注模板数量: {len(rows)}")
    print("-" * 100)
    if not rows:
        print("无已标注项。")
        return

    for row in rows:
        print(
            f"Cluster {str(row['cluster_id']):<6} | "
            f"Label: {str(row['label']):<20} | "
            f"File: {str(row['source_file']):<24} | "
            f"Frame: {row['frame_idx']}"
        )


def export_csv(rows: List[Dict[str, object]], export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cluster_id", "label", "source_file", "frame_idx", "source_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"已导出 CSV: {export_path}")


def main() -> None:
    args = parse_args()
    template_path = Path(args.template_file)
    template_db = load_template_db(template_path)
    rows = collect_labeled_rows(
        template_db,
        include_default_numeric_label=args.include_default_numeric_label,
    )
    print_rows(rows)

    if args.export_csv:
        export_csv(rows, Path(args.export_csv))


if __name__ == "__main__":
    main()

