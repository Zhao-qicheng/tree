import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np


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

LEFT_BONES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT_BONES = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}


def to_display_coords(pose):
    display = np.zeros_like(pose, dtype=np.float32)
    display[:, 0] = pose[:, 0]
    display[:, 1] = pose[:, 2]
    display[:, 2] = -pose[:, 1]
    return display


def html_relative_dir(path, base_dir):
    rel_path = os.path.relpath(path.resolve(), base_dir.resolve())
    return rel_path.replace(os.sep, "/")


def write_image(path, image, quality=90):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def extract_video_frames(video_path, out_dir, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("*.jpg"):
        old_frame.unlink()

    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        write_image(out_dir / f"{count:04d}.jpg", frame)
        count += 1
    cap.release()

    if count == 0:
        raise SystemExit(f"No frames extracted from: {video_path}")
    return count


def count_frame_files(frame_dir):
    return len(list(frame_dir.glob("*.jpg")))


def compact_pose_data(poses):
    rounded = np.round(poses.astype(np.float32), 3)
    return rounded.tolist()


def estimate_radius(poses, percentile=99.0, padding=1.25, min_radius=1.0):
    finite_abs = np.abs(poses[np.isfinite(poses)])
    if finite_abs.size == 0:
        return min_radius
    radius = float(np.percentile(finite_abs, percentile) * padding)
    return max(radius, min_radius)


def estimate_body_height(poses):
    heights = poses[:, :, 2].max(axis=1) - poses[:, :, 2].min(axis=1)
    heights = heights[np.isfinite(heights) & (heights > 1e-6)]
    if heights.size == 0:
        return 0.0
    return float(np.median(heights))


def estimate_skeleton_scale(poses, radius, target_body_height_ratio, max_skeleton_scale):
    body_height = estimate_body_height(poses)
    if body_height <= 0:
        return 1.0
    target_height = 2.0 * radius * target_body_height_ratio
    scale = target_height / body_height
    # The automatic scale should only enlarge small skeletons. Users can still
    # shrink or enlarge it from the HTML controls.
    return float(min(max(scale, 1.0), max_skeleton_scale))


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1114;
      --panel: #181b1f;
      --line: #31373d;
      --text: #eef2f6;
      --muted: #9aa5af;
      --accent: #4da3ff;
      --accent-2: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }
    .app {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-height: 100vh;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #15181b;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
      padding: 12px;
    }
    .viewer {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      min-height: 0;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .viewer-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 14px;
    }
    .image-wrap,
    .canvas-wrap {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 0;
      overflow: hidden;
      background: #f7f8fa;
    }
    img,
    canvas {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 196px);
      object-fit: contain;
      display: block;
    }
    canvas {
      cursor: grab;
      touch-action: none;
      background: #f7f8fa;
    }
    canvas.dragging {
      cursor: grabbing;
    }
    .view-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #15181b;
    }
    .controls {
      display: grid;
      grid-template-columns: auto auto auto minmax(180px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: #15181b;
    }
    button,
    select {
      height: 34px;
      border: 1px solid var(--line);
      background: #20252a;
      color: var(--text);
      padding: 0 12px;
      font: inherit;
      cursor: pointer;
    }
    button:hover,
    select:hover {
      border-color: var(--accent);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .frame-box {
      min-width: 98px;
      color: var(--muted);
      font-size: 14px;
      text-align: right;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .controls { grid-template-columns: repeat(3, auto); }
      .controls input[type="range"] { grid-column: 1 / -1; }
      img,
      canvas { max-height: 48vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>__TITLE__</h1>
      <div class="status">
        <span id="timeLabel">0.000s</span>
        <span id="stateLabel">paused</span>
      </div>
    </header>
    <main>
      <section class="viewer">
        <div class="viewer-title">
          <span>2D H36M Overlay</span>
          <span>same frame index</span>
        </div>
        <div class="image-wrap"><img id="leftImg" alt="2D overlay frame" /></div>
      </section>
      <section class="viewer">
        <div class="viewer-title">
          <span>Interactive 3D Skeleton</span>
          <span class="hint">drag rotate / wheel zoom</span>
        </div>
        <div class="canvas-wrap"><canvas id="scene"></canvas></div>
        <div class="view-tools">
          <button id="viewIso" type="button">Iso</button>
          <button id="viewFront" type="button">Front</button>
          <button id="viewSide" type="button">Side</button>
          <button id="viewTop" type="button">Top</button>
          <button id="viewVideo" type="button">Video-like</button>
          <button id="skeletonBigger" type="button">Skeleton +</button>
          <button id="skeletonSmaller" type="button">Skeleton -</button>
          <button id="zoomIn" type="button">Zoom +</button>
          <button id="zoomOut" type="button">Zoom -</button>
          <span class="hint" id="angleLabel"></span>
        </div>
      </section>
    </main>
    <div class="controls">
      <button id="playPause" type="button">Play</button>
      <button id="prevFrame" type="button">Prev</button>
      <button id="nextFrame" type="button">Next</button>
      <input id="timeline" type="range" min="0" max="__LAST_FRAME__" step="1" value="0" />
      <select id="speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="1.5">1.5x</option>
        <option value="2">2x</option>
      </select>
      <div class="frame-box" id="frameLabel">0 / __LAST_FRAME__</div>
    </div>
  </div>

  <script>
    const poseData = __POSE_DATA__;
    const pairs = __PAIRS__;
    const boneColors = __BONE_COLORS__;
    const leftDir = "__LEFT_DIR__";
    const fps = __FPS__;
    const radius = __RADIUS__;
    const initialSkeletonScale = __SKELETON_SCALE__;
    const lastFrame = poseData.length - 1;

    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const leftImg = document.getElementById("leftImg");
    const playPause = document.getElementById("playPause");
    const prevFrame = document.getElementById("prevFrame");
    const nextFrame = document.getElementById("nextFrame");
    const timeline = document.getElementById("timeline");
    const speed = document.getElementById("speed");
    const frameLabel = document.getElementById("frameLabel");
    const timeLabel = document.getElementById("timeLabel");
    const stateLabel = document.getElementById("stateLabel");
    const angleLabel = document.getElementById("angleLabel");

    let frame = 0;
    let timer = null;
    let yaw = -0.86;
    let pitch = -0.32;
    let zoom = 1.08;
    let skeletonScale = initialSkeletonScale;
    let isDragging = false;
    let lastPointer = { x: 0, y: 0 };

    function padFrame(value) {
      return String(value).padStart(4, "0");
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawScene();
    }

    function rotatePoint(p) {
      const x = p[0] * skeletonScale;
      const y = p[1] * skeletonScale;
      const z = p[2] * skeletonScale;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);

      const x1 = cy * x + sy * y;
      const y1 = -sy * x + cy * y;
      const z1 = z;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return [x1, y2, z2];
    }

    function project(p, rect) {
      const r = rotatePoint(p);
      const size = Math.min(rect.width, rect.height);
      const scale = size * 0.38 * zoom / radius;
      return {
        x: rect.width / 2 + r[0] * scale,
        y: rect.height / 2 - r[2] * scale,
        depth: r[1]
      };
    }

    function line3d(a, b, color, width, alpha, rect) {
      const pa = project(a, rect);
      const pb = project(b, rect);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.restore();
    }

    function drawCube(rect) {
      const r = radius;
      const corners = [
        [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
        [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
      ];
      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
      ];
      const steps = 8;
      for (let i = 0; i <= steps; i += 1) {
        const t = -r + (2 * r * i) / steps;
        line3d([-r, t, -r], [r, t, -r], "#b6bec6", 0.8, 0.42, rect);
        line3d([t, -r, -r], [t, r, -r], "#b6bec6", 0.8, 0.42, rect);
        line3d([-r, r, t], [r, r, t], "#c2c8ce", 0.7, 0.34, rect);
        line3d([t, r, -r], [t, r, r], "#c2c8ce", 0.7, 0.34, rect);
        line3d([-r, -r, t], [-r, r, t], "#c2c8ce", 0.7, 0.30, rect);
        line3d([-r, t, -r], [-r, t, r], "#c2c8ce", 0.7, 0.30, rect);
      }
      for (const [a, b] of edges) {
        line3d(corners[a], corners[b], "#5f6871", 1.6, 0.72, rect);
      }
      line3d([0, 0, 0], [r * 0.72, 0, 0], "#ef4444", 2.0, 0.84, rect);
      line3d([0, 0, 0], [0, r * 0.72, 0], "#2563eb", 2.0, 0.84, rect);
      line3d([0, 0, 0], [0, 0, r * 0.72], "#16a34a", 2.0, 0.84, rect);
    }

    function drawText(text, p, color, rect) {
      const q = project(p, rect);
      ctx.fillStyle = color;
      ctx.font = "13px Arial";
      ctx.fillText(text, q.x + 6, q.y - 6);
    }

    function drawScene() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#f7f8fa";
      ctx.fillRect(0, 0, rect.width, rect.height);

      drawCube(rect);
      const pose = poseData[frame];
      const boneOrder = pairs.map((pair, index) => {
        const a = project(pose[pair[0]], rect);
        const b = project(pose[pair[1]], rect);
        return { pair, index, depth: (a.depth + b.depth) / 2 };
      }).sort((a, b) => b.depth - a.depth);

      for (const item of boneOrder) {
        const [a, b] = item.pair;
        line3d(pose[a], pose[b], boneColors[item.index], 4, 0.95, rect);
      }
      for (const joint of pose) {
        const q = project(joint, rect);
        ctx.beginPath();
        ctx.fillStyle = "#111827";
        ctx.arc(q.x, q.y, 4.2, 0, Math.PI * 2);
        ctx.fill();
      }
      const root = project(pose[0], rect);
      ctx.beginPath();
      ctx.fillStyle = "#f59e0b";
      ctx.arc(root.x, root.y, 5.5, 0, Math.PI * 2);
      ctx.fill();

      drawText("X", [radius * 0.82, 0, 0], "#ef4444", rect);
      drawText("Y", [0, radius * 0.82, 0], "#2563eb", rect);
      drawText("Z", [0, 0, radius * 0.82], "#16a34a", rect);
      angleLabel.textContent = `yaw ${(yaw * 180 / Math.PI).toFixed(0)} deg / pitch ${(pitch * 180 / Math.PI).toFixed(0)} deg / zoom ${zoom.toFixed(2)}x / skeleton ${skeletonScale.toFixed(2)}x`;
    }

    function setFrame(value) {
      frame = Math.max(0, Math.min(lastFrame, value));
      leftImg.src = leftDir + "/" + padFrame(frame) + ".jpg";
      timeline.value = String(frame);
      frameLabel.textContent = `${frame} / ${lastFrame}`;
      timeLabel.textContent = `${(frame / fps).toFixed(3)}s`;
      drawScene();
    }

    function pause() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      playPause.textContent = "Play";
      stateLabel.textContent = "paused";
    }

    function play() {
      if (timer) return;
      playPause.textContent = "Pause";
      stateLabel.textContent = "playing";
      const interval = 1000 / (fps * Number(speed.value));
      timer = setInterval(() => {
        if (frame >= lastFrame) {
          pause();
          return;
        }
        setFrame(frame + 1);
      }, interval);
    }

    function setView(nextYaw, nextPitch, nextZoom = zoom) {
      yaw = nextYaw;
      pitch = Math.max(-1.45, Math.min(1.45, nextPitch));
      zoom = Math.max(0.45, Math.min(3.2, nextZoom));
      drawScene();
    }

    playPause.addEventListener("click", () => {
      if (timer) pause();
      else play();
    });
    prevFrame.addEventListener("click", () => {
      pause();
      setFrame(frame - 1);
    });
    nextFrame.addEventListener("click", () => {
      pause();
      setFrame(frame + 1);
    });
    timeline.addEventListener("input", () => {
      pause();
      setFrame(Number(timeline.value));
    });
    speed.addEventListener("change", () => {
      if (timer) {
        pause();
        play();
      }
    });

    document.getElementById("viewIso").addEventListener("click", () => setView(-0.86, -0.32, 1.08));
    document.getElementById("viewFront").addEventListener("click", () => setView(0, 0, 1.15));
    document.getElementById("viewSide").addEventListener("click", () => setView(-Math.PI / 2, 0, 1.15));
    document.getElementById("viewTop").addEventListener("click", () => setView(-0.6, -1.35, 1.0));
    document.getElementById("viewVideo").addEventListener("click", () => setView(-0.18, -0.18, 1.18));
    document.getElementById("skeletonBigger").addEventListener("click", () => {
      skeletonScale = Math.min(8.0, skeletonScale * 1.15);
      drawScene();
    });
    document.getElementById("skeletonSmaller").addEventListener("click", () => {
      skeletonScale = Math.max(0.25, skeletonScale / 1.15);
      drawScene();
    });
    document.getElementById("zoomIn").addEventListener("click", () => setView(yaw, pitch, zoom * 1.15));
    document.getElementById("zoomOut").addEventListener("click", () => setView(yaw, pitch, zoom / 1.15));

    canvas.addEventListener("pointerdown", (event) => {
      isDragging = true;
      lastPointer = { x: event.clientX, y: event.clientY };
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!isDragging) return;
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      lastPointer = { x: event.clientX, y: event.clientY };
      yaw += dx * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
      drawScene();
    });
    canvas.addEventListener("pointerup", (event) => {
      isDragging = false;
      canvas.classList.remove("dragging");
      canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.08 : 0.92;
      setView(yaw, pitch, zoom * factor);
    }, { passive: false });

    window.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        if (timer) pause();
        else play();
      }
      if (event.code === "ArrowLeft") {
        pause();
        setFrame(frame - 1);
      }
      if (event.code === "ArrowRight") {
        pause();
        setFrame(frame + 1);
      }
    });
    window.addEventListener("resize", resizeCanvas);

    setFrame(0);
    requestAnimationFrame(resizeCanvas);
  </script>
</body>
</html>
"""


def build_bone_colors():
    colors = []
    for pair in H36M_PAIRS:
        key = tuple(pair)
        if key in LEFT_BONES:
            colors.append("#16a34a")
        elif key in RIGHT_BONES:
            colors.append("#dc2626")
        else:
            colors.append("#2563eb")
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-3d-npz", required=True)
    parser.add_argument("--left-frame-dir")
    parser.add_argument("--left-video")
    parser.add_argument("--extract-frame-dir")
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--title", default="Interactive 2D / 3D Skeleton Viewer")
    parser.add_argument("--fps", type=float, default=29.0)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--radius-percentile", type=float, default=99.0)
    parser.add_argument("--radius-padding", type=float, default=1.25)
    parser.add_argument("--skeleton-scale", type=float)
    parser.add_argument("--target-body-height-ratio", type=float, default=0.65)
    parser.add_argument("--max-skeleton-scale", type=float, default=4.0)
    args = parser.parse_args()

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input_3d_npz, allow_pickle=True)
    pred3d = data["pred3d_root_relative_image_units"].astype(np.float32)
    display = np.stack([to_display_coords(pose) for pose in pred3d], axis=0)
    radius = args.radius
    if radius is None:
        radius = estimate_radius(
            display,
            percentile=args.radius_percentile,
            padding=args.radius_padding,
        )
    if args.skeleton_scale is None:
        skeleton_scale = estimate_skeleton_scale(
            display,
            radius,
            target_body_height_ratio=args.target_body_height_ratio,
            max_skeleton_scale=args.max_skeleton_scale,
        )
    else:
        skeleton_scale = args.skeleton_scale

    if args.left_video:
        left_dir = Path(args.extract_frame_dir) if args.extract_frame_dir else out_html.parent / "frames" / "left"
        extracted = extract_video_frames(Path(args.left_video), left_dir, len(display))
        if extracted < len(display):
            print(f"warning: extracted {extracted} frames, but 3D data has {len(display)} frames")
            display = display[:extracted]
    elif args.left_frame_dir:
        left_dir = Path(args.left_frame_dir)
        frame_count = count_frame_files(left_dir)
        if frame_count == 0:
            raise SystemExit(f"No JPG frames found in: {left_dir}")
        if frame_count < len(display):
            print(f"warning: left frames={frame_count}, but 3D data has {len(display)} frames")
            display = display[:frame_count]
    else:
        raise SystemExit("Either --left-video or --left-frame-dir is required")

    html_text = HTML_TEMPLATE
    html_text = html_text.replace("__TITLE__", html.escape(args.title))
    html_text = html_text.replace("__POSE_DATA__", json.dumps(compact_pose_data(display), separators=(",", ":")))
    html_text = html_text.replace("__PAIRS__", json.dumps(H36M_PAIRS, separators=(",", ":")))
    html_text = html_text.replace("__BONE_COLORS__", json.dumps(build_bone_colors(), separators=(",", ":")))
    html_text = html_text.replace("__LEFT_DIR__", html_relative_dir(left_dir, out_html.parent))
    html_text = html_text.replace("__FPS__", f"{args.fps:.8f}")
    html_text = html_text.replace("__RADIUS__", f"{radius:.6f}")
    html_text = html_text.replace("__SKELETON_SCALE__", f"{skeleton_scale:.6f}")
    html_text = html_text.replace("__LAST_FRAME__", str(len(display) - 1))

    out_html.write_text(html_text, encoding="utf-8")
    print(f"frames: {len(display)}")
    print(f"radius: {radius:.3f}")
    print(f"skeleton scale: {skeleton_scale:.3f}")
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
