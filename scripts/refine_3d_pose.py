"""
对 MotionAGFormer 输出的 root-relative 3D 骨架做时序平滑、骨长/对称投影和关节角软约束。

不覆盖原始 3D NPZ/JSON，默认写出 *_ap3d_motionagformer_refined.* 。
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
    refine_3d_poses,
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(description="Refine root-relative 3D poses without overwriting the source files.")
    parser.add_argument("--input-3d-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--vis-out")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_refinement_default.json"))
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()

    cfg = load_refinement_config(args.config)
    fps = args.fps if args.fps is not None else float(cfg.get("fps", 30.0))

    data = np.load(args.input_3d_npz, allow_pickle=True)
    if "pred3d_root_relative_image_units" not in data.files:
        raise SystemExit(f"NPZ missing pred3d_root_relative_image_units: {args.input_3d_npz}")

    poses = np.asarray(data["pred3d_root_relative_image_units"], dtype=np.float32)
    refined, stats = refine_3d_poses(poses, cfg=cfg, fps=fps)

    width = int(np.asarray(data["image_width"]).item()) if "image_width" in data.files else 1
    pred3d_norm = refined / max(width / 2.0, 1e-6)
    pred3d_with_2d_root = refined.copy()
    if "input_2d_pixels" in data.files:
        h36m_2d = np.asarray(data["input_2d_pixels"], dtype=np.float32)
        pred3d_with_2d_root[:, :, :2] = refined[:, :, :2] + h36m_2d[:, 0:1, :2]

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: data[key] for key in data.files}
    payload["pred3d_root_relative_image_units_raw"] = poses
    payload["pred3d_root_relative_image_units"] = refined
    payload["pred3d_norm"] = pred3d_norm.astype(np.float32)
    payload["pred3d_with_2d_root"] = pred3d_with_2d_root.astype(np.float32)
    payload["refinement_config"] = np.asarray(json.dumps(cfg, ensure_ascii=False))
    payload["source_3d_npz"] = str(args.input_3d_npz)
    np.savez_compressed(out_npz, **payload)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "source_3d_npz": str(args.input_3d_npz),
            "format": "h36m_17_root_relative_3d_refined",
            "units": "image_width_scaled_relative_units",
            "joint_names": list(data["h36m_joint_names"]) if "h36m_joint_names" in data.files else list(H36M_NAMES),
            "refinement": {
                "config": cfg,
                "stats": _json_ready(stats),
            },
            "frames": [
                {
                    "frame_id": i,
                    "keypoints_3d": refined[i].tolist(),
                }
                for i in range(len(refined))
            ],
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False)

    if args.vis_out:
        from lift_2d_to_3d_motionagformer_ap3d import draw_3d_video

        vis_out = Path(args.vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        draw_3d_video(refined, vis_out, fps=fps)

    print(f"frames: {stats['frames']}")
    print(f"mean correction: {stats['mean_joint_correction']:.4f}")
    print(f"bone CV: {stats['bone_cv_before']:.4f} -> {stats['bone_cv_after']:.4f}")
    print(f"jerk: {stats['jerk_before']:.4f} -> {stats['jerk_after']:.4f}")
    print(f"saved: {out_npz}")
    if args.out_json:
        print(f"saved: {args.out_json}")
    if args.vis_out:
        print(f"saved: {args.vis_out}")


if __name__ == "__main__":
    main()
