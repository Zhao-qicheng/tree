import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import get_path, get_setting


DEFAULT_VIDEO_NAME = get_setting("PIPELINE_VIDEO_NAME", "test")
DEFAULT_VIDEO_DIR = get_path("PIPELINE_VIDEO_DIR", "input_videos/finefs_test")
DEFAULT_MOTIONAGFORMER_ROOT = get_path(
    "MOTIONAGFORMER_ROOT", "../MotionAGFormer-master"
)
DEFAULT_AP3D_CHECKPOINT = ROOT / "test" / "motionagformer-s-ap3d.pth.tr"
DEFAULT_CONDA_ENV = "mmpose"
DEFAULT_DEVICE = "cpu"
DEFAULT_FPS = 60.0


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def run_command(cmd, stage_name):
    log(f"START {stage_name}")
    print("\n> " + " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd), flush=True)
    start = perf_counter()
    try:
        subprocess.run([str(part) for part in cmd], check=True)
    except subprocess.CalledProcessError as exc:
        log(f"FAILED {stage_name} after {format_seconds(perf_counter() - start)}")
        log(f"Exit code: {exc.returncode}")
        raise
    log(f"DONE {stage_name} after {format_seconds(perf_counter() - start)}")


def python_command(args):
    if args.python_exe:
        return [Path(args.python_exe)]
    if args.no_conda_run:
        return [sys.executable]
    return ["conda", "run", "-n", args.conda_env, "python"]


def require_file(path, label):
    if not Path(path).is_file():
        raise SystemExit(f"{label} not found: {path}")
    log(f"Found {label}: {path}")


def require_dir(path, label):
    if not Path(path).is_dir():
        raise SystemExit(f"{label} not found: {path}")
    log(f"Found {label}: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert corrected 2D H36M JSON to NPZ, lift it to 3D, and rebuild the interactive viewer."
    )
    parser.add_argument("--name", default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--video", default="")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--corrected-json", required=True)
    parser.add_argument("--source-npz", default="")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--motionagformer-root", default=str(DEFAULT_MOTIONAGFORMER_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_AP3D_CHECKPOINT))
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--no-conda-run", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start = perf_counter()

    video_path = Path(args.video) if args.video else Path(args.video_dir) / f"{args.name}.mp4"
    output_name = args.output_name or f"{args.name}_manual"
    source_npz = Path(args.source_npz) if args.source_npz else ROOT / "outputs" / "processed_2d" / f"{args.name}_h36m.npz"
    corrected_json = Path(args.corrected_json)
    motionagformer_root = Path(args.motionagformer_root)
    checkpoint = Path(args.checkpoint)
    mag_config = motionagformer_root / "configs" / "h36m" / "MotionAGFormer-small.yaml"
    py_cmd = python_command(args)

    log("Corrected pose pipeline started")
    log(f"Video: {video_path}")
    log(f"Corrected JSON: {corrected_json}")
    log(f"Output name: {output_name}")
    log(f"Python command: {' '.join(str(part) for part in py_cmd)}")

    require_file(video_path, "Input video")
    require_file(corrected_json, "Corrected 2D JSON")
    require_file(source_npz, "Source 2D NPZ")
    require_dir(motionagformer_root, "MotionAGFormer root")
    require_file(checkpoint, "AP3D checkpoint")
    require_file(mag_config, "MotionAGFormer config")

    corrected_2d_dir = ROOT / "outputs" / "manual_corrected_2d"
    corrected_3d_dir = ROOT / "outputs" / "manual_corrected_3d"
    interactive_dir = ROOT / "outputs" / "interactive_3d"

    corrected_npz = corrected_2d_dir / f"{output_name}_h36m_corrected.npz"
    corrected_json_out = corrected_2d_dir / f"{output_name}_h36m_corrected.json"
    corrected_vis = corrected_2d_dir / f"{output_name}_h36m_corrected_vis.mp4"

    pose3d_npz = corrected_3d_dir / f"{output_name}_ap3d_motionagformer.npz"
    pose3d_json = corrected_3d_dir / f"{output_name}_ap3d_motionagformer.json"
    pose3d_vis = corrected_3d_dir / f"{output_name}_ap3d_motionagformer_vis.mp4"

    interactive_frames = interactive_dir / f"{output_name}_corrected_frames_left"
    interactive_html = interactive_dir / f"{output_name}_corrected_interactive_3d.html"

    run_command(
        py_cmd
        + [
            ROOT / "scripts" / "pose_editor_json_to_npz.py",
            "--corrected-json",
            corrected_json,
            "--source-npz",
            source_npz,
            "--video",
            video_path,
            "--out-npz",
            corrected_npz,
            "--out-json",
            corrected_json_out,
            "--vis-out",
            corrected_vis,
        ],
        "1/3 Corrected JSON to H36M NPZ",
    )

    if not args.skip_3d:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "lift_2d_to_3d_motionagformer_ap3d.py",
                "--input-2d-npz",
                corrected_npz,
                "--checkpoint",
                checkpoint,
                "--motionagformer-root",
                motionagformer_root,
                "--config",
                mag_config,
                "--out-npz",
                pose3d_npz,
                "--out-json",
                pose3d_json,
                "--vis-out",
                pose3d_vis,
                "--device",
                args.device,
            ],
            "2/3 AP3D MotionAGFormer 2D-to-3D lifting",
        )
    else:
        log("SKIP 2/3 AP3D MotionAGFormer 2D-to-3D lifting")

    if not args.skip_viewer:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "build_interactive_3d_viewer.py",
                "--input-3d-npz",
                pose3d_npz,
                "--left-video",
                corrected_vis,
                "--extract-frame-dir",
                interactive_frames,
                "--out-html",
                interactive_html,
                "--title",
                f"{output_name} Corrected 2D / 3D Skeleton",
                "--fps",
                str(args.fps),
            ],
            "3/3 Corrected interactive 3D viewer generation",
        )
    else:
        log("SKIP 3/3 Corrected interactive 3D viewer generation")

    log(f"Corrected pose pipeline complete after {format_seconds(perf_counter() - start)}")
    print(f"\nCorrected 2D NPZ: {corrected_npz}", flush=True)
    print(f"Corrected 2D video: {corrected_vis}", flush=True)
    print(f"Corrected 3D NPZ: {pose3d_npz}", flush=True)
    print(f"Interactive HTML: {interactive_html}", flush=True)


if __name__ == "__main__":
    main()
