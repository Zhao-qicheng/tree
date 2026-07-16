import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VIDEO_NAME = "test1"
DEFAULT_VIDEO_DIR = ROOT / "input_videos" / "finefs_test"
DEFAULT_MMPOSE_ROOT = Path(r"C:\Users\86158\Desktop\mmpose-main")
DEFAULT_MOGANET_ROOT = Path(r"C:\Users\86158\Desktop\MogaNet-main")
DEFAULT_MOTIONAGFORMER_ROOT = Path(r"C:\Users\86158\Desktop\MotionAGFormer-master")
DEFAULT_AP2D_CHECKPOINT = ROOT / "test" / "moganet_b_ap2d_384x288.pth"
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
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    log(f"Found {label}: {path}")


def require_dir(path, label):
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")
    log(f"Found {label}: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the alternate FineFS video -> RTMDet bboxes -> MogaNet AP2D -> H36M -> MotionAGFormer pipeline."
    )
    parser.add_argument("--name", default=DEFAULT_VIDEO_NAME, help="Video stem, for example: test")
    parser.add_argument("--video", help="Full video path. Overrides --name and --video-dir.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="Directory containing <name>.mp4")
    parser.add_argument("--mmpose-root", default=str(DEFAULT_MMPOSE_ROOT))
    parser.add_argument("--moganet-root", default=str(DEFAULT_MOGANET_ROOT))
    parser.add_argument("--motionagformer-root", default=str(DEFAULT_MOTIONAGFORMER_ROOT))
    parser.add_argument("--ap2d-checkpoint", default=str(DEFAULT_AP2D_CHECKPOINT))
    parser.add_argument("--ap3d-checkpoint", default=str(DEFAULT_AP3D_CHECKPOINT))
    parser.add_argument("--det-model", help="Person detector config. Defaults to mmpose demo rtmdet config.")
    parser.add_argument("--det-weights", default=str(DEFAULT_DET_WEIGHTS), help="Person detector checkpoint.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda env name used for child commands")
    parser.add_argument("--python-exe", default="", help="Explicit python executable for child commands")
    parser.add_argument("--no-conda-run", action="store_true", help="Use current python executable for child commands")
    parser.add_argument("--moganet-batch-size", type=int, default=4)
    parser.add_argument("--bbox-thr", type=float, default=0.2)
    parser.add_argument("--limit-frames", type=int, default=0, help="Debug only. 0 means full video.")
    parser.add_argument("--skip-bbox", action="store_true", help="Reuse existing MMPose bbox JSON")
    parser.add_argument("--skip-moganet", action="store_true", help="Reuse existing MogaNet AP2D JSON")
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
    moganet_root = Path(args.moganet_root)
    motionagformer_root = Path(args.motionagformer_root)
    ap2d_checkpoint = Path(args.ap2d_checkpoint)
    ap3d_checkpoint = Path(args.ap3d_checkpoint)
    det_model = Path(args.det_model) if args.det_model else mmpose_root / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
    det_weights = Path(args.det_weights)
    py_cmd = python_command(args)

    log("MogaNet AP2D pipeline started")
    log(f"Video name: {name}")
    log(f"Python command: {' '.join(str(part) for part in py_cmd)}")
    log(f"Device: {args.device}")
    log(f"FPS for viewer: {args.fps}")
    if args.limit_frames and not args.skip_bbox:
        log("--limit-frames applies to the MogaNet AP2D stage; the bbox source stage still processes the input video.")

    log("CHECK input paths")
    require_file(video_path, "Input video")
    require_dir(mmpose_root, "MMPose root")
    require_dir(moganet_root, "MogaNet root")
    require_dir(motionagformer_root, "MotionAGFormer root")
    require_file(ap2d_checkpoint, "AP2D MogaNet checkpoint")
    require_file(ap3d_checkpoint, "AP3D MotionAGFormer checkpoint")
    require_file(det_model, "Person detector config")
    require_file(det_weights, "Person detector checkpoint")

    mmpose_demo = mmpose_root / "demo" / "inferencer_demo.py"
    mag_config = motionagformer_root / "configs" / "h36m" / "MotionAGFormer-small.yaml"
    require_file(mmpose_demo, "MMPose inferencer")
    require_file(mag_config, "MotionAGFormer config")
    log("DONE input path check")

    bbox_pred_dir = ROOT / "outputs" / f"mmpose_pred_{name}"
    bbox_vis_dir = ROOT / "outputs" / f"mmpose_vis_{name}"
    bbox_json = bbox_pred_dir / f"{name}.json"

    moganet_pred_dir = ROOT / "outputs" / f"moganet_pred_{name}"
    moganet_vis_dir = ROOT / "outputs" / f"moganet_vis_{name}"
    moganet_json = moganet_pred_dir / f"{name}.json"
    moganet_vis = moganet_vis_dir / f"{name}.mp4"

    h36m_npz = ROOT / "outputs" / "processed_2d" / f"{name}_moganet_h36m.npz"
    h36m_json = ROOT / "outputs" / "processed_2d" / f"{name}_moganet_h36m.json"
    h36m_vis = ROOT / "outputs" / "processed_2d" / f"{name}_moganet_h36m_vis.mp4"

    pose3d_npz = ROOT / "outputs" / "processed_3d" / f"{name}_moganet_ap3d_motionagformer.npz"
    pose3d_json = ROOT / "outputs" / "processed_3d" / f"{name}_moganet_ap3d_motionagformer.json"
    pose3d_vis = ROOT / "outputs" / "processed_3d" / f"{name}_moganet_ap3d_motionagformer_vis.mp4"

    interactive_dir = ROOT / "outputs" / "interactive_3d"
    interactive_frames = interactive_dir / f"{name}_moganet_frames_left"
    interactive_html = interactive_dir / f"{name}_moganet_interactive_3d.html"

    if not args.skip_bbox:
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
                bbox_pred_dir,
                "--vis-out-dir",
                bbox_vis_dir,
            ],
            "1/5 RTMDet bbox source via MMPose inferencer",
        )
    else:
        log("SKIP 1/5 RTMDet bbox source via MMPose inferencer")
    require_file(bbox_json, "BBox source JSON")
    log(f"Output bbox source JSON: {bbox_json}")

    if not args.skip_moganet:
        moganet_cmd = py_cmd + [
            ROOT / "scripts" / "infer_moganet_ap2d_from_bboxes.py",
            "--video",
            video_path,
            "--bbox-json",
            bbox_json,
            "--out-json",
            moganet_json,
            "--vis-out",
            moganet_vis,
            "--moganet-root",
            moganet_root,
            "--checkpoint",
            ap2d_checkpoint,
            "--device",
            args.device,
            "--bbox-thr",
            str(args.bbox_thr),
            "--batch-size",
            str(args.moganet_batch_size),
        ]
        if args.limit_frames:
            moganet_cmd += ["--limit-frames", str(args.limit_frames)]
        run_command(moganet_cmd, "2/5 MogaNet AP2D 2D pose inference")
    else:
        log("SKIP 2/5 MogaNet AP2D 2D pose inference")
    require_file(moganet_json, "MogaNet AP2D JSON")
    log(f"Output MogaNet AP2D JSON: {moganet_json}")

    if not args.skip_2d:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "select_main_skater_h36m.py",
                "--pred-json",
                moganet_json,
                "--video",
                video_path,
                "--out-npz",
                h36m_npz,
                "--out-json",
                h36m_json,
                "--vis-out",
                h36m_vis,
            ],
            "3/5 Main skater selection and MogaNet H36M conversion",
        )
    else:
        log("SKIP 3/5 Main skater selection and MogaNet H36M conversion")
    require_file(h36m_npz, "MogaNet H36M 2D NPZ")
    require_file(h36m_vis, "MogaNet H36M 2D visualization video")
    log(f"Output MogaNet H36M NPZ: {h36m_npz}")
    log(f"Output MogaNet H36M video: {h36m_vis}")

    if not args.skip_3d:
        run_command(
            py_cmd
            + [
                ROOT / "scripts" / "lift_2d_to_3d_motionagformer_ap3d.py",
                "--input-2d-npz",
                h36m_npz,
                "--checkpoint",
                ap3d_checkpoint,
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
            "4/5 AP3D MotionAGFormer lifting from MogaNet 2D",
        )
    else:
        log("SKIP 4/5 AP3D MotionAGFormer lifting from MogaNet 2D")
    require_file(pose3d_npz, "MogaNet-driven 3D pose NPZ")
    log(f"Output MogaNet-driven 3D NPZ: {pose3d_npz}")

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
                f"{name} MogaNet AP2D / MotionAGFormer 3D",
                "--fps",
                str(args.fps),
            ],
            "5/5 MogaNet interactive 3D viewer generation",
        )
        require_file(interactive_html, "MogaNet interactive HTML")
        log(f"Output MogaNet interactive HTML: {interactive_html}")
    else:
        log("SKIP 5/5 MogaNet interactive 3D viewer generation")

    log(f"MogaNet AP2D pipeline complete after {format_seconds(perf_counter() - pipeline_start)}")
    print("\nDone.", flush=True)
    print(f"MogaNet 3D NPZ: {pose3d_npz}", flush=True)
    if not args.skip_viewer:
        print(f"MogaNet interactive HTML: {interactive_html}", flush=True)


if __name__ == "__main__":
    main()
