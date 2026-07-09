import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


H36M_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

LEFT_JOINTS = {4, 5, 6, 11, 12, 13}
RIGHT_JOINTS = {1, 2, 3, 14, 15, 16}


def to_display_coords(pose):
    """Map model coordinates to a human-readable 3D scene.

    Model output is root-relative. In this AP3D/MotionAGFormer result, image-space
    Y grows downward, so height is represented as -Y.
    """
    display = np.zeros_like(pose, dtype=np.float32)
    display[:, 0] = pose[:, 0]   # left-right
    display[:, 1] = pose[:, 2]   # depth
    display[:, 2] = -pose[:, 1]  # height
    return display


def cube_edges(radius):
    corners = np.array(
        [
            [-radius, -radius, -radius],
            [radius, -radius, -radius],
            [radius, radius, -radius],
            [-radius, radius, -radius],
            [-radius, -radius, radius],
            [radius, -radius, radius],
            [radius, radius, radius],
            [-radius, radius, radius],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return corners, edges


def draw_grid_line(ax, a, b, color="#c8ced4", linewidth=0.55, alpha=0.42):
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        [a[2], b[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )


def draw_cube(ax, radius, grid_steps=8):
    corners, edges = cube_edges(radius)
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="#8a9299",
            linewidth=0.9,
            alpha=0.55,
        )

    ticks = np.linspace(-radius, radius, grid_steps + 1)
    for tick in ticks:
        # Floor plane: X-Y at the lowest height.
        draw_grid_line(ax, (-radius, tick, -radius), (radius, tick, -radius))
        draw_grid_line(ax, (tick, -radius, -radius), (tick, radius, -radius))

        # Back plane: X-Z at far depth.
        draw_grid_line(ax, (-radius, radius, tick), (radius, radius, tick), alpha=0.34)
        draw_grid_line(ax, (tick, radius, -radius), (tick, radius, radius), alpha=0.34)

        # Side plane: Y-Z at left boundary.
        draw_grid_line(ax, (-radius, -radius, tick), (-radius, radius, tick), alpha=0.30)
        draw_grid_line(ax, (-radius, tick, -radius), (-radius, tick, radius), alpha=0.30)


def draw_pose(ax, pose):
    for a, b in H36M_PAIRS:
        color = "#2563eb"
        if a in LEFT_JOINTS or b in LEFT_JOINTS:
            color = "#16a34a"
        if a in RIGHT_JOINTS or b in RIGHT_JOINTS:
            color = "#dc2626"
        ax.plot(
            [pose[a, 0], pose[b, 0]],
            [pose[a, 1], pose[b, 1]],
            [pose[a, 2], pose[b, 2]],
            color=color,
            linewidth=3.0,
        )
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color="#111827", s=18, depthshade=False)
    ax.scatter([0], [0], [0], color="#f59e0b", s=34, depthshade=False)


def render_frame(pose, frame_idx, radius, out_path, dpi=120, grid_steps=8):
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    draw_cube(ax, radius, grid_steps=grid_steps)
    draw_pose(ax, pose)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-62)

    ax.set_xlabel("X  left-right", labelpad=10)
    ax.set_ylabel("Y  depth", labelpad=10)
    ax.set_zlabel("Z  height", labelpad=10)
    ax.set_title(f"AP3D / MotionAGFormer 3D Skeleton  |  frame {frame_idx:04d}", pad=16)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def images_to_video(frame_dir, out_video, fps):
    files = sorted(frame_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No rendered frames found in: {frame_dir}")

    first = read_image(files[0])
    if first is None:
        raise SystemExit(f"Could not read rendered frame: {files[0]}")
    height, width = first.shape[:2]

    # OpenCV on Windows can fail with non-ASCII output paths. Write to an
    # ASCII temp path first, then copy via pathlib/shutil.
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for file in files:
        image = read_image(file)
        if image is None:
            continue
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(image)
    writer.release()
    shutil.copyfile(temp_path, out_video)
    temp_path.unlink(missing_ok=True)


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-3d-npz", required=True)
    parser.add_argument("--out-video", required=True)
    parser.add_argument("--out-frame-dir", required=True)
    parser.add_argument("--fps", type=float, default=29.0)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--grid-steps", type=int, default=8)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    data = np.load(args.input_3d_npz, allow_pickle=True)
    pred3d = data["pred3d_root_relative_image_units"].astype(np.float32)
    display = np.stack([to_display_coords(pose) for pose in pred3d], axis=0)

    if args.radius is None:
        radius = float(np.max(np.abs(display)) * 1.18)
        radius = max(radius, 1.0)
    else:
        radius = args.radius

    frame_dir = Path(args.out_frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("*.jpg"):
        old_frame.unlink()

    for idx, pose in enumerate(display):
        render_frame(
            pose,
            idx,
            radius,
            frame_dir / f"{idx:04d}.jpg",
            grid_steps=args.grid_steps,
        )

    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    images_to_video(frame_dir, out_video, args.fps)

    if not args.keep_frames:
        for frame in frame_dir.glob("*.jpg"):
            frame.unlink()

    print(f"frames: {len(display)}")
    print(f"cube radius: {radius:.3f}")
    print(f"saved: {out_video}")
    if args.keep_frames:
        print(f"frames saved in: {frame_dir}")


if __name__ == "__main__":
    main()
