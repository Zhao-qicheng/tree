"""将 RTMW3D WholeBody-133 融合到 MotionAGFormer H36M-17 核心骨架。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.extended_pose_fusion import (  # noqa: E402
    build_extended_npz_payload,
    fuse_extended_sequence,
    load_extended_config,
)
from utils.extended_pose_schema import SCHEMA_VERSION  # noqa: E402


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(description="Fuse RTMW3D 133-point extras onto a refined H36M-17 3D skeleton.")
    parser.add_argument("--input-3d-npz", required=True, help="MotionAGFormer / refined 17-point 3D NPZ")
    parser.add_argument("--input-wholebody-npz", required=True, help="Raw RTMW3D 133-point NPZ")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--config", default=str(ROOT / "configs" / "extended_pose_default.json"))
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()

    cfg = load_extended_config(args.config)
    fps = args.fps if args.fps is not None else float(cfg.get("fps", 60.0))

    core_data = np.load(args.input_3d_npz, allow_pickle=True)
    if "pred3d_root_relative_image_units" not in core_data.files:
        raise SystemExit(f"3D NPZ missing pred3d_root_relative_image_units: {args.input_3d_npz}")
    core = np.asarray(core_data["pred3d_root_relative_image_units"], dtype=np.float64)

    whole_data = np.load(args.input_wholebody_npz, allow_pickle=True)
    if "keypoints_3d_cam" not in whole_data.files:
        raise SystemExit(f"WholeBody NPZ missing keypoints_3d_cam: {args.input_wholebody_npz}")
    whole = np.asarray(whole_data["keypoints_3d_cam"], dtype=np.float64)
    scores = np.asarray(whole_data["keypoint_scores"], dtype=np.float64)
    selected = np.asarray(whole_data["selected"], dtype=bool) if "selected" in whole_data.files else np.ones((len(whole),), dtype=bool)

    frame_count = min(len(core), len(whole), len(scores))
    if len(core) != len(whole):
        print(f"warning: core frames={len(core)} wholebody frames={len(whole)}; using first {frame_count}")
    core = core[:frame_count]
    whole = whole[:frame_count]
    scores = scores[:frame_count]
    selected = selected[:frame_count]
    scores = np.where(selected[:, None], scores, 0.0)

    fused = fuse_extended_sequence(core, whole, scores, cfg=cfg, fps=fps)
    payload = build_extended_npz_payload(
        core_data,
        fused,
        extra_meta={
            "source_3d_npz": str(args.input_3d_npz),
            "source_wholebody_npz": str(args.input_wholebody_npz),
            "schema_version": SCHEMA_VERSION,
        },
    )
    # 对齐截断后的核心数组
    payload["pred3d_root_relative_image_units"] = fused["core_pose_3d"]
    payload["core_pose_3d"] = fused["core_pose_3d"]

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_3d_npz": str(args.input_3d_npz),
                    "source_wholebody_npz": str(args.input_wholebody_npz),
                    "format": SCHEMA_VERSION,
                    "stats": _json_ready(fused["stats"]),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"saved: {out_json}")

    print(f"frames: {frame_count}")
    print(f"accepted frames: {fused['stats']['accepted_frames']}")
    print(f"valid extended mean: {fused['stats']['valid_extended_mean']:.3f}")
    print(f"saved: {out_npz}")


if __name__ == "__main__":
    main()
