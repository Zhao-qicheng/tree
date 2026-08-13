"""比较原始与精修姿态序列，输出 JSON 质量报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.pose_constraints import compare_pose_sequences, load_refinement_config  # noqa: E402


def _load_sequence(path: str, mode: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if mode == "2d":
        key = "h36m_keypoints"
    else:
        key = "pred3d_root_relative_image_units"
    if key not in data.files:
        raise SystemExit(f"{path} missing {key}")
    return np.asarray(data[key])


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs refined pose NPZ files.")
    parser.add_argument("--before", required=True, help="Original NPZ")
    parser.add_argument("--after", required=True, help="Refined NPZ")
    parser.add_argument("--mode", choices=("2d", "3d"), required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_refinement_default.json"))
    args = parser.parse_args()

    cfg = load_refinement_config(args.config)
    fps = args.fps if args.fps is not None else float(cfg.get("fps", 30.0))
    before = _load_sequence(args.before, args.mode)
    after = _load_sequence(args.after, args.mode)
    report = compare_pose_sequences(before, after, fps=fps, mode=args.mode)
    report["before"] = str(Path(args.before))
    report["after"] = str(Path(args.after))
    report["fps"] = fps
    if args.mode == "3d":
        report["constraint_correction"] = report["mean_correction"]

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text, encoding="utf-8")
        print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
