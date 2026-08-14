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
DEFAULT_SOURCE_3D_NPZ = ROOT / "outputs" / "processed_3d" / f"{DEFAULT_VIDEO_NAME}_ap3d_motionagformer.npz"
DEFAULT_LEFT_VIDEO = ROOT / "outputs" / "processed_2d" / f"{DEFAULT_VIDEO_NAME}_h36m_vis.mp4"
DEFAULT_CONDA_ENV = "mmpose"
DEFAULT_DEVICE = "cpu"
DEFAULT_FPS = 30.0


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
        description="Convert corrected 3D JSON to NPZ and rebuild the interactive 3D viewer."
    )
    parser.add_argument("--name", default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--video", default="")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--source-3d-npz", default=str(DEFAULT_SOURCE_3D_NPZ))
    parser.add_argument("--corrected-json", required=True)
    parser.add_argument("--left-video", default=str(DEFAULT_LEFT_VIDEO))
    parser.add_argument("--output-name", default="")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Reserved for command compatibility. This 3D correction step does not run a model.",
    )
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--no-conda-run", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start = perf_counter()

    video_path = Path(args.video) if args.video else Path(args.video_dir) / f"{args.name}.mp4"
    output_name = args.output_name or f"{args.name}_manual3d"
    source_3d_npz = Path(args.source_3d_npz)
    corrected_json = Path(args.corrected_json)
    left_video = Path(args.left_video) if args.left_video else None
    py_cmd = python_command(args)

    log("Manual 3D correction pipeline started")
    log(f"Video: {video_path}")
    log(f"Source 3D NPZ: {source_3d_npz}")
    log(f"Corrected JSON: {corrected_json}")
    log(f"Output name: {output_name}")
    log(f"Python command: {' '.join(str(part) for part in py_cmd)}")

    require_file(video_path, "Input video")
    require_file(source_3d_npz, "Source 3D NPZ")
    require_file(corrected_json, "Corrected 3D JSON")
    if left_video:
        require_file(left_video, "Left video")

    corrected_3d_dir = ROOT / "outputs" / "manual_corrected_3d"
    interactive_dir = ROOT / "outputs" / "interactive_3d"

    corrected_npz = corrected_3d_dir / f"{output_name}_corrected_3d.npz"
    corrected_json_out = corrected_3d_dir / f"{output_name}_corrected_3d.json"
    corrected_vis = corrected_3d_dir / f"{output_name}_corrected_3d_vis.mp4"
    interactive_frames = interactive_dir / f"{output_name}_frames_left"
    interactive_html = interactive_dir / f"{output_name}_interactive_3d.html"

    run_command(
        py_cmd
        + [
            ROOT / "scripts" / "pose3d_editor_json_to_npz.py",
            "--corrected-json",
            corrected_json,
            "--source-npz",
            source_3d_npz,
            "--out-npz",
            corrected_npz,
            "--out-json",
            corrected_json_out,
            "--vis-out",
            corrected_vis,
            "--video",
            video_path,
        ],
        "1/2 Corrected 3D JSON to NPZ",
    )

    if not args.skip_viewer:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "build_interactive_3d_viewer.py",
                "--input-3d-npz",
                corrected_npz,
                "--left-video",
                corrected_vis if corrected_vis.exists() else (left_video or corrected_vis),
                "--extract-frame-dir",
                interactive_frames,
                "--out-html",
                interactive_html,
                "--title",
                f"{output_name} Corrected 3D Skeleton",
                "--fps",
                str(args.fps),
            ],
            "2/2 Interactive 3D viewer generation",
        )
    else:
        log("SKIP 2/2 Interactive 3D viewer generation")

    log(f"Manual 3D correction pipeline complete after {format_seconds(perf_counter() - start)}")
    print(f"\nCorrected 3D NPZ: {corrected_npz}", flush=True)
    print(f"Corrected 3D JSON: {corrected_json_out}", flush=True)
    print(f"Corrected 3D video: {corrected_vis}", flush=True)
    print(f"Interactive HTML: {interactive_html}", flush=True)


if __name__ == "__main__":
    main()
