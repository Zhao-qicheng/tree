import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults if you prefer running this script without command args.
DEFAULT_VIDEO_NAME = "test1"
DEFAULT_VIDEO_DIR = ROOT / "input_videos" / "finefs_test"
DEFAULT_MMPOSE_ROOT = Path(r"C:\Users\86158\Desktop\mmpose-main")
DEFAULT_MOTIONAGFORMER_ROOT = Path(r"C:\Users\86158\Desktop\MotionAGFormer-master")
DEFAULT_AP3D_CHECKPOINT = ROOT / "test" / "motionagformer-s-ap3d.pth.tr"
DEFAULT_DET_WEIGHTS = (
    Path.home()
    / ".cache"
    / "torch"
    / "hub"
    / "checkpoints"
    / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)
DEFAULT_FPS = 25
DEFAULT_DEVICE = "cpu"
DEFAULT_CONDA_ENV = "mmpose"


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
    print("\n> " + " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))
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
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    log(f"Found {label}: {path}")


def require_dir(path, label):
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")
    log(f"Found {label}: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full FineFS video -> 2D pose -> H36M -> 3D pose -> interactive viewer pipeline."
    )
    parser.add_argument("--name", default=DEFAULT_VIDEO_NAME, help="Video stem, for example: test")
    parser.add_argument("--video", help="Full video path. Overrides --name and --video-dir.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="Directory containing <name>.mp4")
    parser.add_argument("--mmpose-root", default=str(DEFAULT_MMPOSE_ROOT))
    parser.add_argument("--motionagformer-root", default=str(DEFAULT_MOTIONAGFORMER_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_AP3D_CHECKPOINT))
    parser.add_argument("--det-model", help="Person detector config. Defaults to mmpose demo rtmdet config.")
    parser.add_argument("--det-weights", default=str(DEFAULT_DET_WEIGHTS), help="Person detector checkpoint.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda env name used for child commands")
    parser.add_argument("--python-exe", default="", help="Explicit python executable for child commands")
    parser.add_argument("--no-conda-run", action="store_true", help="Use current python executable for child commands")
    parser.add_argument("--skip-mmpose", action="store_true", help="Reuse existing MMPose JSON")
    parser.add_argument("--skip-2d", action="store_true", help="Reuse existing H36M 2D output")
    parser.add_argument("--skip-3d", action="store_true", help="Reuse existing 3D output")
    parser.add_argument("--skip-viewer", action="store_true", help="Do not generate interactive HTML")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start = perf_counter()

    video_path = Path(args.video) if args.video else Path(args.video_dir) / f"{args.name}.mp4"
    name = video_path.stem
    mmpose_root = Path(args.mmpose_root)
    motionagformer_root = Path(args.motionagformer_root)
    checkpoint = Path(args.checkpoint)
    det_model = Path(args.det_model) if args.det_model else mmpose_root / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
    det_weights = Path(args.det_weights)
    py_cmd = python_command(args)

    log("Pipeline started")
    log(f"Video name: {name}")
    log(f"Python command: {' '.join(str(part) for part in py_cmd)}")
    log(f"Device: {args.device}")
    log(f"FPS for viewer: {args.fps}")

    log("CHECK input paths")
    require_file(video_path, "Input video")
    require_dir(mmpose_root, "MMPose root")
    require_dir(motionagformer_root, "MotionAGFormer root")
    require_file(checkpoint, "AP3D checkpoint")
    require_file(det_model, "Person detector config")
    require_file(det_weights, "Person detector checkpoint")

    mmpose_demo = mmpose_root / "demo" / "inferencer_demo.py"
    mag_config = motionagformer_root / "configs" / "h36m" / "MotionAGFormer-small.yaml"
    require_file(mmpose_demo, "MMPose inferencer")
    require_file(mag_config, "MotionAGFormer config")
    log("DONE input path check")

    pred_dir = ROOT / "outputs" / f"mmpose_pred_{name}"
    vis_dir = ROOT / "outputs" / f"mmpose_vis_{name}"
    pred_json = pred_dir / f"{name}.json"

    h36m_npz = ROOT / "outputs" / "processed_2d" / f"{name}_h36m.npz"
    h36m_json = ROOT / "outputs" / "processed_2d" / f"{name}_h36m.json"
    h36m_vis = ROOT / "outputs" / "processed_2d" / f"{name}_h36m_vis.mp4"

    pose3d_npz = ROOT / "outputs" / "processed_3d" / f"{name}_ap3d_motionagformer.npz"
    pose3d_json = ROOT / "outputs" / "processed_3d" / f"{name}_ap3d_motionagformer.json"
    pose3d_vis = ROOT / "outputs" / "processed_3d" / f"{name}_ap3d_motionagformer_vis.mp4"

    interactive_dir = ROOT / "outputs" / "interactive_3d"
    interactive_frames = interactive_dir / f"{name}_frames_left"
    interactive_html = interactive_dir / f"{name}_interactive_3d.html"

    if not args.skip_mmpose:
        run_command(
            py_cmd
            + [
                mmpose_demo,
                video_path,
                "--pose2d",
                "body",
                "--device",
                args.device,
                "--det-model",
                det_model,
                "--det-weights",
                det_weights,
                "--det-cat-ids",
                "0",
                "--show-progress",
                "--pred-out-dir",
                pred_dir,
                "--vis-out-dir",
                vis_dir,
            ],
            "1/4 MMPose 2D pose inference",
        )
    else:
        log("SKIP 1/4 MMPose 2D pose inference")
    require_file(pred_json, "MMPose prediction JSON")
    log(f"Output MMPose JSON: {pred_json}")

    if not args.skip_2d:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "select_main_skater_h36m.py",
                "--pred-json",
                pred_json,
                "--video",
                video_path,
                "--out-npz",
                h36m_npz,
                "--out-json",
                h36m_json,
                "--vis-out",
                h36m_vis,
            ],
            "2/4 Main skater selection and H36M conversion",
        )
    else:
        log("SKIP 2/4 Main skater selection and H36M conversion")
    require_file(h36m_npz, "H36M 2D NPZ")
    require_file(h36m_vis, "H36M 2D visualization video")
    log(f"Output H36M NPZ: {h36m_npz}")
    log(f"Output H36M video: {h36m_vis}")

    if not args.skip_3d:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "lift_2d_to_3d_motionagformer_ap3d.py",
                "--input-2d-npz",
                h36m_npz,
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
            "3/4 AP3D MotionAGFormer 2D-to-3D lifting",
        )
    else:
        log("SKIP 3/4 AP3D MotionAGFormer 2D-to-3D lifting")
    require_file(pose3d_npz, "3D pose NPZ")
    log(f"Output 3D NPZ: {pose3d_npz}")

    if not args.skip_viewer:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "build_interactive_3d_viewer.py",
                "--input-3d-npz",
                pose3d_npz,
                "--left-video",
                h36m_vis,
                "--extract-frame-dir",
                interactive_frames,
                "--out-html",
                interactive_html,
                "--title",
                f"{name} Interactive 2D / 3D Skeleton",
                "--fps",
                str(args.fps),
            ],
            "4/4 Interactive 3D viewer generation",
        )
        require_file(interactive_html, "Interactive HTML")
        log(f"Output interactive HTML: {interactive_html}")
    else:
        log("SKIP 4/4 Interactive 3D viewer generation")

    log(f"Pipeline complete after {format_seconds(perf_counter() - pipeline_start)}")
    print("\nDone.", flush=True)
    print(f"3D NPZ: {pose3d_npz}", flush=True)
    print(f"Interactive HTML: {interactive_html}", flush=True)


if __name__ == "__main__":
    main()
