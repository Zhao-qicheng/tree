"""
对 H36M-17 二维关键点做低置信度修复、跳点检测、短缺口插值和 One Euro 滤波。

不覆盖原始 NPZ/JSON，默认写出 *_h36m_refined.* 。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from utils.pose_constraints import (  # noqa: E402
    H36M_NAMES,
    load_refinement_config,
    refine_2d_keypoints,
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Refine H36M 2D keypoints without overwriting the source files.")
    parser.add_argument("--input-2d-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--vis-out")
    parser.add_argument("--video", help="Optional source video used only for visualization.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_refinement_default.json"))
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()

    cfg = load_refinement_config(args.config)
    fps = args.fps if args.fps is not None else float(cfg.get("fps", 30.0))

    data = np.load(args.input_2d_npz, allow_pickle=True)
    if "h36m_keypoints" not in data.files:
        raise SystemExit(f"NPZ missing h36m_keypoints: {args.input_2d_npz}")
    keypoints = np.asarray(data["h36m_keypoints"], dtype=np.float32)
    refined, stats = refine_2d_keypoints(keypoints, cfg=cfg, fps=fps)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: data[key] for key in data.files}
    payload["h36m_keypoints"] = refined
    payload["h36m_keypoints_raw"] = keypoints
    payload["refinement_config"] = np.asarray(json.dumps(cfg, ensure_ascii=False))
    payload["source_2d_npz"] = str(args.input_2d_npz)
    np.savez_compressed(out_npz, **payload)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        frame_ids = data["frame_ids"] if "frame_ids" in data.files else np.arange(len(refined))
        selected = data["selected_indices"] if "selected_indices" in data.files else np.full((len(refined),), -1)
        json_payload = {
            "source_2d_npz": str(args.input_2d_npz),
            "format": "h36m_17_refined",
            "joint_names": list(data["h36m_joint_names"]) if "h36m_joint_names" in data.files else list(H36M_NAMES),
            "refinement": {
                "config": cfg,
                "stats": _json_ready(stats),
            },
            "frames": [
                {
                    "frame_id": int(frame_ids[i]),
                    "selected_instance": int(selected[i]),
                    "keypoints": refined[i].tolist(),
                }
                for i in range(len(refined))
            ],
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False)

    if args.vis_out:
        if not args.video:
            raise SystemExit("--vis-out requires --video")
        from select_main_skater_h36m import draw_h36m_video

        vis_out = Path(args.vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        draw_h36m_video(args.video, refined, vis_out)

    print(f"frames: {stats['frames']}")
    print(f"low-score: {stats['low_score_count']}")
    print(f"speed outliers: {stats['speed_outlier_count']}")
    print(f"repaired: {stats['repaired_count']}")
    print(f"saved: {out_npz}")
    if args.out_json:
        print(f"saved: {args.out_json}")
    if args.vis_out:
        print(f"saved: {args.vis_out}")


if __name__ == "__main__":
    main()
