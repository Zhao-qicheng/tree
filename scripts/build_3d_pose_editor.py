import argparse
import html
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import get_setting


DEFAULT_VIDEO_NAME = get_setting("PIPELINE_VIDEO_NAME", "test")
DEFAULT_3D_NPZ = ROOT / "outputs" / "processed_3d" / f"{DEFAULT_VIDEO_NAME}_ap3d_motionagformer.npz"
DEFAULT_LEFT_VIDEO = ROOT / "outputs" / "processed_2d" / f"{DEFAULT_VIDEO_NAME}_h36m_vis.mp4"
DEFAULT_LEFT_FRAME_DIR = ROOT / "outputs" / "pose3d_editor_frames" / DEFAULT_VIDEO_NAME
DEFAULT_OUT_HTML = ROOT / "outputs" / "pose3d_editor" / f"{DEFAULT_VIDEO_NAME}_3d_pose_editor.html"
DEFAULT_TITLE = f"{DEFAULT_VIDEO_NAME} 3D Pose Editor"


H36M_NAMES = [
    "root",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "nose",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

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
      --edited: #ec4899;
      --warn: #f59e0b;
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
      grid-template-columns: minmax(0, 1fr) minmax(0, 1.55fr) 340px;
      gap: 12px;
      min-height: 0;
      padding: 12px;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      min-width: 0;
      min-height: 0;
    }
    .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 14px;
    }
    .reference-wrap {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .reference-body {
      position: relative;
      overflow: hidden;
      background: #0b0d0f;
      display: grid;
      place-items: center;
    }
    .reference-body img {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 196px);
      object-fit: contain;
      display: block;
    }
    .editor-wrap {
      display: grid;
      grid-template-rows: minmax(0, 1.45fr) minmax(0, 1fr);
      gap: 12px;
      min-width: 0;
      min-height: 0;
    }
    .orbit-body {
      position: relative;
      overflow: hidden;
      background: #f7f8fa;
    }
    .orbit-body canvas,
    .ortho-body canvas {
      width: 100%;
      height: 100%;
      display: block;
      touch-action: none;
      cursor: crosshair;
      background: #f7f8fa;
    }
    .orbit-body canvas.dragging,
    .ortho-body canvas.dragging {
      cursor: grabbing;
    }
    .ortho-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      min-width: 0;
      min-height: 0;
    }
    .ortho-card {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      min-height: 0;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .controls {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 10px;
      min-height: 0;
    }
    .block {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px;
      min-width: 0;
    }
    .grid2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .grid3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    button,
    select,
    input[type="number"] {
      min-height: 34px;
      border: 1px solid var(--line);
      background: #20252a;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover, select:hover, input[type="number"]:hover {
      border-color: var(--accent);
    }
    button.primary {
      border-color: #2563eb;
      background: #1d4ed8;
    }
    button.warn {
      border-color: #a16207;
      background: #713f12;
    }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    select { width: 100%; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    input[type="number"] { width: 100%; }
    .joint-list {
      overflow: auto;
      border: 1px solid var(--line);
      background: #111417;
      min-height: 0;
    }
    .joint {
      display: grid;
      grid-template-columns: 22px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 7px 8px;
      border-bottom: 1px solid #252a2f;
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
    }
    .joint:last-child { border-bottom: 0; }
    .joint.active {
      color: var(--text);
      background: #243044;
    }
    .joint.edited {
      color: #f4b3d6;
      background: #2a1730;
    }
    .joint.edited.active {
      background: #3a2341;
    }
    .joint .dot {
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: #5ca1ff;
    }
    .joint.edited .dot { background: var(--edited); }
    .score {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .joint.edited .score { color: #f4b3d6; }
    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .timeline {
      display: grid;
      grid-template-columns: auto auto auto minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #15181b;
    }
    .frame-box {
      min-width: 120px;
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    @media (max-width: 1260px) {
      main { grid-template-columns: 1fr; }
      .ortho-grid { grid-template-columns: 1fr; }
      .timeline { grid-template-columns: repeat(3, auto); }
      .timeline input[type="range"] { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>__TITLE__</h1>
      <div class="status">
        <span id="stateLabel">paused</span>
        <span id="timeLabel">0.000s</span>
        <span id="editedLabel">0 edits</span>
      </div>
    </header>

    <main>
      <section class="panel reference-wrap">
        <div class="panel-title">
          <span>Reference Video</span>
          <span id="refFrameLabel">frame 0</span>
        </div>
        <div class="reference-body">
          <img id="refImg" alt="reference frame" />
        </div>
      </section>

      <section class="editor-wrap">
        <section class="panel">
          <div class="panel-title">
            <span>3D Orbit Preview</span>
            <span class="small">drag rotate / wheel zoom / select in list</span>
          </div>
          <div class="orbit-body">
            <canvas id="orbitCanvas"></canvas>
          </div>
        </section>
        <section class="ortho-grid">
          <section class="ortho-card">
            <div class="panel-title"><span>Front (X / Height)</span><span class="small">drag joint</span></div>
            <div class="ortho-body"><canvas id="frontCanvas"></canvas></div>
          </section>
          <section class="ortho-card">
            <div class="panel-title"><span>Side (Depth / Height)</span><span class="small">drag joint</span></div>
            <div class="ortho-body"><canvas id="sideCanvas"></canvas></div>
          </section>
          <section class="ortho-card">
            <div class="panel-title"><span>Top (X / Depth)</span><span class="small">drag joint</span></div>
            <div class="ortho-body"><canvas id="topCanvas"></canvas></div>
          </section>
        </section>
      </section>

      <aside class="controls">
        <section class="block">
          <div class="grid2">
            <button id="saveBtn" class="primary" type="button">Save JSON</button>
            <button id="downloadBtn" type="button">Download</button>
            <button id="undoBtn" type="button">Undo</button>
            <button id="resetFrameBtn" class="warn" type="button">Reset Frame</button>
          </div>
        </section>

        <section class="block">
          <div class="row">
            <select id="jointSelect"></select>
          </div>
          <div class="grid2" style="margin-top:8px;">
            <button id="copyPrevJointBtn" type="button">Copy Prev Joint</button>
            <button id="copyNextJointBtn" type="button">Copy Next Joint</button>
            <button id="copyPrevFrameBtn" type="button">Copy Prev Frame</button>
            <button id="copyNextFrameBtn" type="button">Copy Next Frame</button>
          </div>
          <div class="grid2" style="margin-top:8px;">
            <button id="interpJointBtn" type="button">Interp Joint</button>
            <button id="interpAllBtn" type="button">Interp All</button>
            <button id="prevEditedBtn" type="button">Prev Edit</button>
            <button id="nextEditedBtn" type="button">Next Edit</button>
          </div>
        </section>

        <section class="block">
          <div class="grid3">
            <label class="small">X<input id="coordX" type="number" step="0.1" /></label>
            <label class="small">Depth<input id="coordDepth" type="number" step="0.1" /></label>
            <label class="small">Height<input id="coordHeight" type="number" step="0.1" /></label>
          </div>
          <div class="grid2" style="margin-top:8px;">
            <input id="nudgeStep" type="number" step="0.1" min="0.1" value="5" />
            <button id="applyCoordsBtn" class="primary" type="button">Apply XYZ</button>
          </div>
          <div class="grid3" style="margin-top:8px;">
            <button data-axis="0" data-delta="-1" class="nudge-btn" type="button">X -</button>
            <button data-axis="0" data-delta="1" class="nudge-btn" type="button">X +</button>
            <button data-axis="1" data-delta="-1" class="nudge-btn" type="button">D -</button>
            <button data-axis="1" data-delta="1" class="nudge-btn" type="button">D +</button>
            <button data-axis="2" data-delta="-1" class="nudge-btn" type="button">H -</button>
            <button data-axis="2" data-delta="1" class="nudge-btn" type="button">H +</button>
          </div>
          <div class="grid2" style="margin-top:8px;">
            <button id="viewIso" type="button">Iso</button>
            <button id="viewFront" type="button">Front</button>
            <button id="viewSide" type="button">Side</button>
            <button id="viewTop" type="button">Top</button>
            <button id="viewVideo" type="button">Video-like</button>
            <button id="viewReset" type="button">Reset View</button>
            <button id="skeletonBigger" type="button">Skeleton +</button>
            <button id="skeletonSmaller" type="button">Skeleton -</button>
            <button id="zoomIn" type="button">Zoom +</button>
            <button id="zoomOut" type="button">Zoom -</button>
          </div>
        </section>

        <section class="joint-list" id="jointList"></section>
      </aside>
    </main>

    <div class="timeline">
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
    const payload = __PAYLOAD__;
    const leftDir = "__LEFT_DIR__";
    const fps = Number(payload.fps || 30);
    let frames = JSON.parse(JSON.stringify(payload.frames));
    let original = JSON.parse(JSON.stringify(frames.map((f) => f.keypoints_3d_display)));
    let edited = frames.map(() => Array(payload.joint_names.length).fill(false));
    const poseData = frames.map((f) => f.keypoints_3d_display);
    const pairs = payload.pairs;
    const boneColors = payload.bone_colors;
    const jointNames = payload.joint_names;
    const lastFrame = frames.length - 1;
    const radius = Number(payload.radius);
    const initialSkeletonScale = Number(payload.skeleton_scale);

    const orbitCanvas = document.getElementById("orbitCanvas");
    const frontCanvas = document.getElementById("frontCanvas");
    const sideCanvas = document.getElementById("sideCanvas");
    const topCanvas = document.getElementById("topCanvas");
    const refImg = document.getElementById("refImg");
    const playPause = document.getElementById("playPause");
    const prevFrame = document.getElementById("prevFrame");
    const nextFrame = document.getElementById("nextFrame");
    const timeline = document.getElementById("timeline");
    const speed = document.getElementById("speed");
    const frameLabel = document.getElementById("frameLabel");
    const refFrameLabel = document.getElementById("refFrameLabel");
    const timeLabel = document.getElementById("timeLabel");
    const stateLabel = document.getElementById("stateLabel");
    const editedLabel = document.getElementById("editedLabel");
    const jointList = document.getElementById("jointList");
    const jointSelect = document.getElementById("jointSelect");
    const coordX = document.getElementById("coordX");
    const coordDepth = document.getElementById("coordDepth");
    const coordHeight = document.getElementById("coordHeight");
    const nudgeStep = document.getElementById("nudgeStep");
    const undoBtn = document.getElementById("undoBtn");

    const canvases = {
      orbit: orbitCanvas,
      front: frontCanvas,
      side: sideCanvas,
      top: topCanvas,
    };

    const ctxs = {
      orbit: orbitCanvas.getContext("2d"),
      front: frontCanvas.getContext("2d"),
      side: sideCanvas.getContext("2d"),
      top: topCanvas.getContext("2d"),
    };

    const viewDefs = {
      orbit: { kind: "orbit" },
      front: { kind: "ortho", axes: [0, 2], title: "Front" },
      side: { kind: "ortho", axes: [1, 2], title: "Side" },
      top: { kind: "ortho", axes: [0, 1], title: "Top" },
    };

    let frame = 0;
    let selectedJoint = 0;
    let timer = null;
    let yaw = -0.86;
    let pitch = -0.32;
    let zoom = 1.08;
    let skeletonScale = initialSkeletonScale;
    let history = [];
    let dragState = null;

    function padFrame(value) {
      return String(value).padStart(4, "0");
    }

    function currentPose() {
      return frames[frame].keypoints_3d_display;
    }

    function currentJoint() {
      return currentPose()[selectedJoint];
    }

    function resizeCanvas(canvas, drawFn) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawFn();
    }

    function resizeAll() {
      resizeCanvas(orbitCanvas, drawOrbit);
      resizeCanvas(frontCanvas, () => drawOrtho("front"));
      resizeCanvas(sideCanvas, () => drawOrtho("side"));
      resizeCanvas(topCanvas, () => drawOrtho("top"));
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

    function projectOrbit(point, rect) {
      const r = rotatePoint(point);
      const size = Math.min(rect.width, rect.height);
      const scale = size * 0.38 * zoom / radius;
      return { x: rect.width / 2 + r[0] * scale, y: rect.height / 2 - r[2] * scale, depth: r[1] };
    }

    function projectOrtho(point, axes, rect) {
      const size = Math.min(rect.width, rect.height);
      const scale = size * 0.38 * zoom * skeletonScale / radius;
      return {
        x: rect.width / 2 + point[axes[0]] * scale,
        y: rect.height / 2 - point[axes[1]] * scale,
        scale,
      };
    }

    function line(ctx, a, b, color, width, alpha) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }

    function drawGrid(ctx, rect, scale) {
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const half = radius * scale;
      ctx.save();
      ctx.strokeStyle = "#c8ced4";
      ctx.lineWidth = 0.8;
      ctx.globalAlpha = 0.38;
      for (let i = -4; i <= 4; i += 1) {
        const t = (i / 4) * half;
        ctx.beginPath();
        ctx.moveTo(centerX - half, centerY + t);
        ctx.lineTo(centerX + half, centerY + t);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(centerX + t, centerY - half);
        ctx.lineTo(centerX + t, centerY + half);
        ctx.stroke();
      }
      ctx.strokeStyle = "#87919b";
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.6;
      ctx.strokeRect(centerX - half, centerY - half, half * 2, half * 2);
      ctx.restore();
    }

    function drawOrbit() {
      const canvas = orbitCanvas;
      const ctx = ctxs.orbit;
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#f7f8fa";
      ctx.fillRect(0, 0, rect.width, rect.height);
      drawCube(ctx, rect);
      const pose = currentPose();
      const order = pairs.map((pair, index) => {
        const pa = projectOrbit(pose[pair[0]], rect);
        const pb = projectOrbit(pose[pair[1]], rect);
        return { pair, index, depth: (pa.depth + pb.depth) / 2 };
      }).sort((a, b) => b.depth - a.depth);
      for (const item of order) {
        const [a, b] = item.pair;
        line(ctx, projectOrbit(pose[a], rect), projectOrbit(pose[b], rect), boneColors[item.index], 4, 0.95);
      }
      for (let i = 0; i < pose.length; i += 1) {
        const q = projectOrbit(pose[i], rect);
        ctx.beginPath();
        ctx.fillStyle = i === selectedJoint ? "#fff7c2" : edited[frame][i] ? "#ec4899" : "#111827";
        ctx.strokeStyle = i === selectedJoint ? "#f59e0b" : "#ffffff";
        ctx.lineWidth = i === selectedJoint ? 3 : 1.5;
        ctx.arc(q.x, q.y, i === selectedJoint ? 7 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.fillStyle = "#1f2937";
      ctx.font = "13px Arial";
      ctx.fillText("X", 16, rect.height - 16);
      ctx.fillText("Depth", 40, rect.height - 16);
      ctx.fillText("Height", rect.width - 72, 18);
    }

    function drawCube(ctx, rect) {
      const r = radius * skeletonScale;
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
        line3d(ctx, rect, [-r, t, -r], [r, t, -r], "#b6bec6", 0.8, 0.42);
        line3d(ctx, rect, [t, -r, -r], [t, r, -r], "#b6bec6", 0.8, 0.42);
        line3d(ctx, rect, [-r, r, t], [r, r, t], "#c2c8ce", 0.7, 0.34);
        line3d(ctx, rect, [t, r, -r], [t, r, r], "#c2c8ce", 0.7, 0.34);
        line3d(ctx, rect, [-r, -r, t], [-r, r, t], "#c2c8ce", 0.7, 0.30);
        line3d(ctx, rect, [-r, t, -r], [-r, t, r], "#c2c8ce", 0.7, 0.30);
      }
      for (const [a, b] of edges) {
        line3d(ctx, rect, corners[a], corners[b], "#5f6871", 1.6, 0.72);
      }
      line3d(ctx, rect, [0, 0, 0], [r * 0.72, 0, 0], "#ef4444", 2.0, 0.84);
      line3d(ctx, rect, [0, 0, 0], [0, r * 0.72, 0], "#2563eb", 2.0, 0.84);
      line3d(ctx, rect, [0, 0, 0], [0, 0, r * 0.72], "#16a34a", 2.0, 0.84);
    }

    function line3d(ctx, rect, a, b, color, width, alpha) {
      const pa = projectOrbit(a, rect);
      const pb = projectOrbit(b, rect);
      line(ctx, pa, pb, color, width, alpha);
    }

    function drawOrtho(name) {
      const def = viewDefs[name];
      const canvas = canvases[name];
      const ctx = ctxs[name];
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#f7f8fa";
      ctx.fillRect(0, 0, rect.width, rect.height);
      const pose = currentPose();
      const axes = def.axes;
      const scale = Math.min(rect.width, rect.height) * 0.38 * zoom * skeletonScale / radius;
      drawGrid(ctx, rect, scale);

      const order = pairs.map((pair, index) => {
        const pa = projectOrtho(pose[pair[0]], axes, rect);
        const pb = projectOrtho(pose[pair[1]], axes, rect);
        const hiddenAxis = 3 - axes[0] - axes[1];
        return { pair, index, depth: (pose[pair[0]][hiddenAxis] + pose[pair[1]][hiddenAxis]) / 2 };
      }).sort((a, b) => a.depth - b.depth);

      for (const item of order) {
        const [a, b] = item.pair;
        line(ctx, projectOrtho(pose[a], axes, rect), projectOrtho(pose[b], axes, rect), boneColors[item.index], 3.2, 0.95);
      }
      for (let i = 0; i < pose.length; i += 1) {
        const q = projectOrtho(pose[i], axes, rect);
        ctx.beginPath();
        ctx.fillStyle = edited[frame][i] ? "#ec4899" : "#111827";
        ctx.strokeStyle = i === selectedJoint ? "#fef08a" : "#ffffff";
        ctx.lineWidth = i === selectedJoint ? 3 : 1.5;
        ctx.arc(q.x, q.y, i === selectedJoint ? 7 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.fillStyle = "#1f2937";
      ctx.font = "13px Arial";
      ctx.fillText(def.axes[0] === 0 ? "X" : def.axes[0] === 1 ? "Depth" : "Height", 16, 18);
      ctx.fillText(def.axes[1] === 0 ? "X" : def.axes[1] === 1 ? "Depth" : "Height", rect.width - 88, rect.height - 14);
    }

    function updateReference() {
      refImg.src = leftDir + "/" + padFrame(frame) + ".jpg";
      refFrameLabel.textContent = `frame ${frame}`;
    }

    function updateEditedLabel() {
      const count = edited.reduce((sum, row) => sum + row.filter(Boolean).length, 0);
      editedLabel.textContent = `${count} edits`;
    }

    function updateCoordInputs() {
      const p = currentJoint();
      coordX.value = Number(p[0]).toFixed(3);
      coordDepth.value = Number(p[1]).toFixed(3);
      coordHeight.value = Number(p[2]).toFixed(3);
    }

    function updateJointPanel() {
      for (const el of jointList.children) {
        const j = Number(el.dataset.joint);
        el.classList.toggle("active", j === selectedJoint);
        el.classList.toggle("edited", edited[frame][j]);
        el.querySelector(".score").textContent = "";
      }
      jointSelect.value = String(selectedJoint);
      updateCoordInputs();
    }

    function drawAll() {
      drawOrbit();
      drawOrtho("front");
      drawOrtho("side");
      drawOrtho("top");
    }

    function setFrame(value) {
      frame = Math.max(0, Math.min(lastFrame, Math.round(value)));
      timeline.value = String(frame);
      frameLabel.textContent = `${frame} / ${lastFrame}`;
      timeLabel.textContent = `${(frame / fps).toFixed(3)}s`;
      updateReference();
      updateCoordInputs();
      updateEditedLabel();
      drawAll();
    }

    function selectJoint(index) {
      selectedJoint = Math.max(0, Math.min(jointNames.length - 1, index));
      updateJointPanel();
      drawAll();
    }

    function pushHistory(action) {
      history.push(action);
      if (history.length > 200) history.shift();
      undoBtn.disabled = history.length === 0;
    }

    function markEdited(joint) {
      edited[frame][joint] = true;
      updateEditedLabel();
      updateJointPanel();
      drawAll();
    }

    function setJointCoords(frameIndex, joint, coords, mark = true) {
      const before = [...frames[frameIndex].keypoints_3d_display[joint]];
      const editedBefore = edited[frameIndex][joint];
      frames[frameIndex].keypoints_3d_display[joint] = [coords[0], coords[1], coords[2]];
      if (mark) {
        edited[frameIndex][joint] = true;
      }
      return { before, editedBefore };
    }

    function applyCoordsFromInputs() {
      const coords = [
        Number(coordX.value),
        Number(coordDepth.value),
        Number(coordHeight.value),
      ];
      if (coords.some((v) => !Number.isFinite(v))) return;
      const snapshot = setJointCoords(frame, selectedJoint, coords, true);
      pushHistory({ type: "joint", frame, joint: selectedJoint, ...snapshot });
      updateCoordInputs();
      drawAll();
    }

    function copyJointFromFrame(sourceFrame) {
      if (sourceFrame < 0 || sourceFrame > lastFrame) return;
      const snapshot = setJointCoords(frame, selectedJoint, frames[sourceFrame].keypoints_3d_display[selectedJoint], true);
      pushHistory({ type: "joint", frame, joint: selectedJoint, ...snapshot });
      updateCoordInputs();
      drawAll();
    }

    function copyFrameFromFrame(sourceFrame) {
      if (sourceFrame < 0 || sourceFrame > lastFrame) return;
      const before = JSON.parse(JSON.stringify(frames[frame].keypoints_3d_display));
      const editedBefore = [...edited[frame]];
      frames[frame].keypoints_3d_display = JSON.parse(JSON.stringify(frames[sourceFrame].keypoints_3d_display));
      edited[frame] = Array(jointNames.length).fill(true);
      pushHistory({ type: "frame", frame, before, editedBefore });
      updateCoordInputs();
      updateEditedLabel();
      drawAll();
    }

    function resetFrame() {
      const before = JSON.parse(JSON.stringify(frames[frame].keypoints_3d_display));
      const editedBefore = [...edited[frame]];
      frames[frame].keypoints_3d_display = JSON.parse(JSON.stringify(original[frame]));
      edited[frame] = Array(jointNames.length).fill(false);
      pushHistory({ type: "frame", frame, before, editedBefore });
      updateCoordInputs();
      updateEditedLabel();
      drawAll();
    }

    function editedFrameIndices() {
      const out = [];
      for (let i = 0; i <= lastFrame; i += 1) {
        if (edited[i].some(Boolean)) out.push(i);
      }
      return out;
    }

    function interpolateJoint(joint) {
      const keys = [];
      for (let i = 0; i <= lastFrame; i += 1) {
        if (edited[i][joint]) keys.push(i);
      }
      if (keys.length < 2) return;
      const before = frames.map((f) => [...f.keypoints_3d_display[joint]]);
      const editedBefore = edited.map((row) => row[joint]);
      for (let k = 0; k < keys.length - 1; k += 1) {
        const a = keys[k];
        const b = keys[k + 1];
        const pa = frames[a].keypoints_3d_display[joint];
        const pb = frames[b].keypoints_3d_display[joint];
        for (let i = a + 1; i < b; i += 1) {
          const t = (i - a) / (b - a);
          frames[i].keypoints_3d_display[joint] = [
            pa[0] * (1 - t) + pb[0] * t,
            pa[1] * (1 - t) + pb[1] * t,
            pa[2] * (1 - t) + pb[2] * t,
          ];
          edited[i][joint] = true;
        }
      }
      pushHistory({ type: "joint_series", joint, before, editedBefore });
      updateCoordInputs();
      updateEditedLabel();
      drawAll();
    }

    function applyViewToJoint(viewName, x, y, recordHistory = false) {
      const canvas = canvases[viewName];
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width, rect.height) * 0.38 * zoom * skeletonScale / radius;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const def = viewDefs[viewName];
      const coords = [...currentJoint()];
      coords[def.axes[0]] = (x - centerX) / scale;
      coords[def.axes[1]] = -(y - centerY) / scale;
      const snapshot = setJointCoords(frame, selectedJoint, coords, true);
      if (recordHistory) {
        pushHistory({ type: "joint", frame, joint: selectedJoint, ...snapshot });
      }
      updateCoordInputs();
      updateEditedLabel();
      drawAll();
    }

    function hitTestJoint(viewName, x, y) {
      const canvas = canvases[viewName];
      const rect = canvas.getBoundingClientRect();
      const pose = currentPose();
      let best = null;
      let bestDist = Infinity;
      const threshold = viewName === "orbit" ? 18 : 16;
      for (let i = 0; i < pose.length; i += 1) {
        const q = viewName === "orbit"
          ? projectOrbit(pose[i], rect)
          : projectOrtho(pose[i], viewDefs[viewName].axes, rect);
        const d = Math.hypot(q.x - x, q.y - y);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      return bestDist <= threshold ? best : null;
    }

    function makeExportPayload() {
      const manualEdits = [];
      for (let i = 0; i <= lastFrame; i += 1) {
        for (let j = 0; j < jointNames.length; j += 1) {
          if (edited[i][j]) {
            manualEdits.push({
              frame_index: i,
              frame_id: frames[i].frame_id,
              joint_index: j,
              joint_name: jointNames[j],
              keypoint_3d_display: frames[i].keypoints_3d_display[j],
            });
          }
        }
      }
      return {
        source_3d_npz: payload.source_3d_npz,
        source_left_video: payload.source_left_video,
        source_2d_npz: payload.source_2d_npz,
        correction_format: "h36m_17_manual_3d_display_v1",
        coordinate_space: "display_x_depth_height",
        units: "image_width_scaled_relative_units",
        image_width: payload.image_width,
        image_height: payload.image_height,
        fps,
        joint_names,
        pairs,
        manual_edits: manualEdits,
        frames,
      };
    }

    async function saveJson(preferPicker) {
      const text = JSON.stringify(makeExportPayload(), null, 2);
      const suggestedName = `${payload.output_stem}_corrected_3d.json`;
      if (preferPicker && window.showSaveFilePicker) {
        const handle = await window.showSaveFilePicker({
          suggestedName,
          types: [{ description: "JSON", accept: { "application/json": [".json"] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
        return;
      }
      const blob = new Blob([text], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = suggestedName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function buildJointControls() {
      for (let i = 0; i < jointNames.length; i += 1) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `${i} ${jointNames[i]}`;
        jointSelect.appendChild(opt);

        const item = document.createElement("div");
        item.className = "joint";
        item.dataset.joint = String(i);
        item.innerHTML = `<span class="dot"></span><span>${i} ${jointNames[i]}</span><span class="score"></span>`;
        item.addEventListener("click", () => selectJoint(i));
        jointList.appendChild(item);
      }
    }

    function undo() {
      const item = history.pop();
      if (!item) return;
      if (item.type === "joint_series") {
        for (let i = 0; i <= lastFrame; i += 1) {
          frames[i].keypoints_3d_display[item.joint] = [...item.before[i]];
          edited[i][item.joint] = item.editedBefore[i];
        }
      } else if (item.type === "frame") {
        frames[item.frame].keypoints_3d_display = JSON.parse(JSON.stringify(item.before));
        edited[item.frame] = [...item.editedBefore];
      } else {
        frames[item.frame].keypoints_3d_display[item.joint] = [...item.before];
        edited[item.frame][item.joint] = item.editedBefore;
      }
      updateCoordInputs();
      updateEditedLabel();
      drawAll();
      undoBtn.disabled = history.length === 0;
    }

    function play() {
      if (timer) return;
      playPause.textContent = "Pause";
      stateLabel.textContent = "playing";
      const interval = Math.max(16, 1000 / (fps * Number(speed.value)));
      timer = setInterval(() => {
        if (frame >= lastFrame) {
          pause();
          return;
        }
        setFrame(frame + 1);
      }, interval);
    }

    function pause() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      playPause.textContent = "Play";
      stateLabel.textContent = "paused";
    }

    function setView(nextYaw, nextPitch, nextZoom = zoom) {
      yaw = nextYaw;
      pitch = Math.max(-1.45, Math.min(1.45, nextPitch));
      zoom = Math.max(0.45, Math.min(3.2, nextZoom));
      drawAll();
    }

    function findNextEditedFrame(start) {
      for (let i = start; i <= lastFrame; i += 1) {
        if (edited[i].some(Boolean)) return i;
      }
      return null;
    }

    function findPrevEditedFrame(start) {
      for (let i = start; i >= 0; i -= 1) {
        if (edited[i].some(Boolean)) return i;
      }
      return null;
    }

    function syncViewSelection(viewName, event) {
      const rect = canvases[viewName].getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = hitTestJoint(viewName, x, y);
      if (hit !== null) {
        selectJoint(hit);
      }
    }

    orbitCanvas.addEventListener("pointerdown", (event) => {
      const rect = orbitCanvas.getBoundingClientRect();
      dragState = {
        view: "orbit",
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        moved: false,
      };
      orbitCanvas.classList.add("dragging");
      orbitCanvas.setPointerCapture(event.pointerId);
      syncViewSelection("orbit", event);
    });
    orbitCanvas.addEventListener("pointermove", (event) => {
      if (!dragState || dragState.view !== "orbit") return;
      const dx = event.clientX - dragState.lastX;
      const dy = event.clientY - dragState.lastY;
      if (Math.abs(event.clientX - dragState.startX) + Math.abs(event.clientY - dragState.startY) > 4) {
        dragState.moved = true;
      }
      dragState.lastX = event.clientX;
      dragState.lastY = event.clientY;
      yaw += dx * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
      drawOrbit();
    });
    orbitCanvas.addEventListener("pointerup", (event) => {
      orbitCanvas.classList.remove("dragging");
      orbitCanvas.releasePointerCapture(event.pointerId);
      dragState = null;
    });
    orbitCanvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.08 : 0.92;
      setView(yaw, pitch, zoom * factor);
    }, { passive: false });

    for (const name of ["front", "side", "top"]) {
      canvases[name].addEventListener("pointerdown", (event) => {
        pause();
        const rect = canvases[name].getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const hit = hitTestJoint(name, x, y);
        if (hit !== null) selectJoint(hit);
        dragState = {
          view: name,
          startX: x,
          startY: y,
          lastX: x,
          lastY: y,
          moved: false,
          before: [...currentJoint()],
          editedBefore: edited[frame][selectedJoint],
        };
        canvases[name].classList.add("dragging");
        canvases[name].setPointerCapture(event.pointerId);
        applyViewToJoint(name, x, y);
      });
      canvases[name].addEventListener("pointermove", (event) => {
        if (!dragState || dragState.view !== name) return;
        const rect = canvases[name].getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        dragState.lastX = x;
        dragState.lastY = y;
        applyViewToJoint(name, x, y);
      });
      canvases[name].addEventListener("pointerup", (event) => {
        if (dragState && dragState.view === name) {
          pushHistory({ type: "joint", frame, joint: selectedJoint, before: dragState.before, editedBefore: dragState.editedBefore });
        }
        canvases[name].classList.remove("dragging");
        canvases[name].releasePointerCapture(event.pointerId);
        dragState = null;
      });
    }

    document.getElementById("saveBtn").addEventListener("click", () => saveJson(true).catch((err) => alert(err.message)));
    document.getElementById("downloadBtn").addEventListener("click", () => saveJson(false));
    document.getElementById("undoBtn").addEventListener("click", undo);
    document.getElementById("resetFrameBtn").addEventListener("click", resetFrame);
    document.getElementById("interpJointBtn").addEventListener("click", () => interpolateJoint(selectedJoint));
    document.getElementById("interpAllBtn").addEventListener("click", () => {
      for (let j = 0; j < jointNames.length; j += 1) interpolateJoint(j);
    });
    document.getElementById("prevEditedBtn").addEventListener("click", () => {
      const prev = findPrevEditedFrame(frame - 1);
      if (prev !== null) setFrame(prev);
    });
    document.getElementById("nextEditedBtn").addEventListener("click", () => {
      const next = findNextEditedFrame(frame + 1);
      if (next !== null) setFrame(next);
    });
    document.getElementById("copyPrevJointBtn").addEventListener("click", () => copyJointFromFrame(frame - 1));
    document.getElementById("copyNextJointBtn").addEventListener("click", () => copyJointFromFrame(frame + 1));
    document.getElementById("copyPrevFrameBtn").addEventListener("click", () => copyFrameFromFrame(frame - 1));
    document.getElementById("copyNextFrameBtn").addEventListener("click", () => copyFrameFromFrame(frame + 1));
    document.getElementById("applyCoordsBtn").addEventListener("click", applyCoordsFromInputs);
    for (const btn of document.querySelectorAll(".nudge-btn")) {
      btn.addEventListener("click", () => {
        const axis = Number(btn.dataset.axis);
        const delta = Number(btn.dataset.delta);
        const step = Number(nudgeStep.value) || 1;
        const coords = [...currentJoint()];
        coords[axis] += delta * step;
        const snapshot = setJointCoords(frame, selectedJoint, coords, true);
        pushHistory({ type: "joint", frame, joint: selectedJoint, ...snapshot });
        updateCoordInputs();
        updateEditedLabel();
        drawAll();
      });
    }
    jointSelect.addEventListener("change", () => selectJoint(Number(jointSelect.value)));
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
    playPause.addEventListener("click", () => {
      if (timer) pause();
      else play();
    });
    prevFrame.addEventListener("click", () => { pause(); setFrame(frame - 1); });
    nextFrame.addEventListener("click", () => { pause(); setFrame(frame + 1); });
    window.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        if (timer) pause();
        else play();
      } else if (event.code === "ArrowLeft") {
        event.preventDefault();
        pause();
        setFrame(frame - 1);
      } else if (event.code === "ArrowRight") {
        event.preventDefault();
        pause();
        setFrame(frame + 1);
      }
    });
    window.addEventListener("resize", resizeAll);
    for (const btn of [
      document.getElementById("viewIso"),
      document.getElementById("viewFront"),
      document.getElementById("viewSide"),
      document.getElementById("viewTop"),
      document.getElementById("viewVideo"),
      document.getElementById("viewReset"),
      document.getElementById("skeletonBigger"),
      document.getElementById("skeletonSmaller"),
      document.getElementById("zoomIn"),
      document.getElementById("zoomOut"),
    ]) {
      btn.addEventListener("click", () => {
        if (btn.id === "viewIso") setView(-0.86, -0.32, 1.08);
        else if (btn.id === "viewFront") setView(0, 0, 1.15);
        else if (btn.id === "viewSide") setView(-Math.PI / 2, 0, 1.15);
        else if (btn.id === "viewTop") setView(-0.6, -1.35, 1.0);
        else if (btn.id === "viewVideo") setView(-0.18, -0.18, 1.18);
        else if (btn.id === "viewReset") setView(-0.86, -0.32, 1.08);
        else if (btn.id === "skeletonBigger") { skeletonScale = Math.min(8.0, skeletonScale * 1.15); drawAll(); }
        else if (btn.id === "skeletonSmaller") { skeletonScale = Math.max(0.25, skeletonScale / 1.15); drawAll(); }
        else if (btn.id === "zoomIn") setView(yaw, pitch, zoom * 1.15);
        else if (btn.id === "zoomOut") setView(yaw, pitch, zoom / 1.15);
      });
    }
    coordX.addEventListener("change", applyCoordsFromInputs);
    coordDepth.addEventListener("change", applyCoordsFromInputs);
    coordHeight.addEventListener("change", applyCoordsFromInputs);

    buildJointControls();
    undoBtn.disabled = true;
    setFrame(0);
    resizeAll();
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
    parser.add_argument("--input-3d-npz", default=str(DEFAULT_3D_NPZ))
    parser.add_argument("--left-video", default=str(DEFAULT_LEFT_VIDEO))
    parser.add_argument("--left-frame-dir", default="")
    parser.add_argument("--extract-frame-dir", default=str(DEFAULT_LEFT_FRAME_DIR))
    parser.add_argument("--out-html", default=str(DEFAULT_OUT_HTML))
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--radius-percentile", type=float, default=99.0)
    parser.add_argument("--radius-padding", type=float, default=1.25)
    parser.add_argument("--skeleton-scale", type=float)
    parser.add_argument("--target-body-height-ratio", type=float, default=0.65)
    parser.add_argument("--max-skeleton-scale", type=float, default=4.0)
    args = parser.parse_args()

    input_3d_npz = Path(args.input_3d_npz)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(input_3d_npz, allow_pickle=True)
    pred3d = data["pred3d_root_relative_image_units"].astype(np.float32)
    display = np.stack([to_display_coords(pose) for pose in pred3d], axis=0)
    radius = args.radius
    if radius is None:
        radius = estimate_radius(display, percentile=args.radius_percentile, padding=args.radius_padding)
    if args.skeleton_scale is None:
        skeleton_scale = estimate_skeleton_scale(
            display,
            radius,
            target_body_height_ratio=args.target_body_height_ratio,
            max_skeleton_scale=args.max_skeleton_scale,
        )
    else:
        skeleton_scale = args.skeleton_scale

    left_video = Path(args.left_video) if args.left_video else None
    left_frame_dir = Path(args.left_frame_dir) if args.left_frame_dir else None
    if left_video and left_video.is_file():
        extracted_dir = Path(args.extract_frame_dir) if args.extract_frame_dir else out_html.parent / "frames" / "left"
        extracted = extract_video_frames(left_video, extracted_dir, len(display))
        if extracted < len(display):
            print(f"warning: extracted {extracted} frames, but 3D data has {len(display)} frames")
            display = display[:extracted]
            pred3d = pred3d[:extracted]
    elif left_frame_dir and left_frame_dir.is_dir():
        frame_count = count_frame_files(left_frame_dir)
        if frame_count == 0:
            raise SystemExit(f"No JPG frames found in: {left_frame_dir}")
        if frame_count < len(display):
            print(f"warning: left frames={frame_count}, but 3D data has {len(display)} frames")
            display = display[:frame_count]
            pred3d = pred3d[:frame_count]
        extracted_dir = left_frame_dir
    else:
        raise SystemExit("Either --left-video or --left-frame-dir is required")

    fps = args.fps
    if fps is None:
        fps = float(data["fps"]) if "fps" in data.files else 29.0

    html_text = HTML_TEMPLATE
    html_text = html_text.replace("__TITLE__", html.escape(args.title))
    html_text = html_text.replace(
        "__PAYLOAD__",
        json.dumps(
            {
                "source_3d_npz": str(input_3d_npz),
                "source_left_video": str(left_video) if left_video else "",
                "source_2d_npz": str(data["source_2d_npz"]) if "source_2d_npz" in data.files else "",
                "output_stem": out_html.stem,
                "coordinate_space": "display_x_depth_height",
                "units": "image_width_scaled_relative_units",
                "image_width": int(np.asarray(data["image_width"]).item()) if "image_width" in data.files else 0,
                "image_height": int(np.asarray(data["image_height"]).item()) if "image_height" in data.files else 0,
                "fps": fps,
                "radius": radius,
                "skeleton_scale": skeleton_scale,
                "joint_names": H36M_NAMES,
                "pairs": H36M_PAIRS,
                "bone_colors": build_bone_colors(),
                "frames": [
                    {
                        "frame_id": int(i),
                        "keypoints_3d_display": display[i].tolist(),
                    }
                    for i in range(len(display))
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    html_text = html_text.replace("__LEFT_DIR__", html_relative_dir(extracted_dir, out_html.parent))
    html_text = html_text.replace("__LAST_FRAME__", str(len(display) - 1))

    out_html.write_text(html_text, encoding="utf-8")
    print(f"frames: {len(display)}")
    print(f"radius: {radius:.3f}")
    print(f"skeleton scale: {skeleton_scale:.3f}")
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
