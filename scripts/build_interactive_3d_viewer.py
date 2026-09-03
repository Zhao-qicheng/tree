import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGION_MODEL_BUNDLE = ROOT / "region_model" / "dist" / "human17-region-model.js"


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


def read_region_model_bundle():
    if not REGION_MODEL_BUNDLE.is_file():
        raise RuntimeError(
            "Human17 region-model bundle is missing. Run `npm install` and `npm run build` in region_model/."
        )
    return REGION_MODEL_BUNDLE.read_text(encoding="utf-8")


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


def compact_extended_payload(poses, valid):
    poses = np.asarray(poses, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    pose_out = []
    valid_out = []
    for frame_pose, frame_valid in zip(poses, valid):
        joints = []
        flags = []
        for point, is_valid in zip(frame_pose, frame_valid):
            if is_valid and np.all(np.isfinite(point)):
                joints.append([round(float(point[0]), 3), round(float(point[1]), 3), round(float(point[2]), 3)])
                flags.append(True)
            else:
                joints.append(None)
                flags.append(False)
        pose_out.append(joints)
        valid_out.append(flags)
    return pose_out, valid_out


def compact_scores(scores):
    values = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(np.clip(values, 0.0, 1.0), 3).tolist()


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
    .toggle-control,
    .range-control {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 34px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .toggle-control input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }
    .range-control input[type="range"] {
      width: 112px;
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
      .app { display: block; }
      header {
        align-items: flex-start;
        flex-direction: column;
        gap: 6px;
      }
      .status {
        flex-wrap: wrap;
        white-space: normal;
      }
      main { grid-template-columns: 1fr; }
      .viewer { grid-template-rows: auto auto; }
      .viewer-title {
        align-items: flex-start;
        flex-direction: column;
        gap: 4px;
      }
      .image-wrap { aspect-ratio: 16 / 9; }
      .canvas-wrap { aspect-ratio: 1 / 1; }
      .view-tools .range-control {
        flex: 1 1 230px;
        min-width: 0;
      }
      .view-tools .range-control input[type="range"] {
        flex: 1 1 auto;
        min-width: 80px;
      }
      #angleLabel {
        flex: 1 0 100%;
        overflow-wrap: anywhere;
      }
      .controls { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .controls button,
      .controls select {
        width: 100%;
        padding-inline: 8px;
      }
      .controls input[type="range"] { grid-column: 1 / -1; }
      .frame-box {
        min-width: 0;
        grid-column: 2 / -1;
      }
      img,
      canvas { max-height: none; }
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
          <label class="toggle-control" title="Show or hide the capsule region model">
            <input id="regionToggle" type="checkbox" checked />
            <span>Region model</span>
          </label>
          <label class="toggle-control" title="Show or hide the original Human17 line skeleton">
            <input id="skeletonToggle" type="checkbox" checked />
            <span>Skeleton</span>
          </label>
          <label class="toggle-control extended-only" title="Show oriented head, hands and feet from extra keypoints">
            <input id="extendedToggle" type="checkbox" checked />
            <span>Extended extras</span>
          </label>
          <label class="toggle-control extended-only" title="Show oriented feet">
            <input id="feetToggle" type="checkbox" checked />
            <span>Feet</span>
          </label>
          <label class="toggle-control extended-only" title="Show simplified hands and fingertips">
            <input id="handsToggle" type="checkbox" checked />
            <span>Hands</span>
          </label>
          <label class="toggle-control extended-only" title="Show face visor used to read head facing">
            <input id="faceToggle" type="checkbox" checked />
            <span>Face</span>
          </label>
          <label class="range-control node-overlay-control" title="Show RTMW3D diagnostic points and links">
            <span>RTMW3D nodes</span>
            <select id="nodeMode">
              <option value="hidden">Hidden</option>
              <option value="selected">Selected 22</option>
              <option value="full">Full 133</option>
            </select>
          </label>
          <label class="range-control node-overlay-control" title="Minimum RTMW3D confidence">
            <span>Node confidence</span>
            <input id="nodeThreshold" type="range" min="0" max="1" step="0.05" value="__NODE_THRESHOLD__" />
          </label>
          <span class="hint node-overlay-control" id="nodeCount">RTMW3D nodes: 0</span>
          <label class="range-control" title="Adjust capsule and joint thickness">
            <span>Thickness</span>
            <input id="regionThickness" type="range" min="0.65" max="1.55" step="0.05" value="1" />
          </label>
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
__REGION_MODEL_BUNDLE__
  </script>

  <script>
    const poseData = __POSE_DATA__;
    const extendedPoseData = __EXTENDED_POSE_DATA__;
    const extendedValid = __EXTENDED_VALID__;
    const extendedScores = __EXTENDED_SCORES__;
    const hasExtended = __HAS_EXTENDED__;
    const wholebodyPoseData = __WHOLEBODY_POSE_DATA__;
    const wholebodyValid = __WHOLEBODY_VALID__;
    const wholebodyScores = __WHOLEBODY_SCORES__;
    const hasWholebody = __HAS_WHOLEBODY__;
    const initialNodeMode = "__NODE_MODE__";
    const pairs = __PAIRS__;
    const boneColors = __BONE_COLORS__;
    const leftDir = "__LEFT_DIR__";
    const fps = __FPS__;
    const radius = __RADIUS__;
    const initialSkeletonScale = __SKELETON_SCALE__;
    const lastFrame = poseData.length - 1;

    const canvas = document.getElementById("scene");
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
    const regionToggle = document.getElementById("regionToggle");
    const skeletonToggle = document.getElementById("skeletonToggle");
    const regionThickness = document.getElementById("regionThickness");
    const extendedToggle = document.getElementById("extendedToggle");
    const feetToggle = document.getElementById("feetToggle");
    const handsToggle = document.getElementById("handsToggle");
    const faceToggle = document.getElementById("faceToggle");
    const nodeMode = document.getElementById("nodeMode");
    const nodeThreshold = document.getElementById("nodeThreshold");
    const nodeCount = document.getElementById("nodeCount");

    if (!hasExtended) {
      document.querySelectorAll(".extended-only").forEach((node) => {
        node.style.display = "none";
      });
    }
    if (!hasExtended && !hasWholebody) {
      document.querySelectorAll(".node-overlay-control").forEach((node) => {
        node.style.display = "none";
      });
    }
    nodeMode.value = initialNodeMode;
    nodeMode.querySelector('option[value="selected"]').disabled = !hasExtended;
    nodeMode.querySelector('option[value="full"]').disabled = !hasWholebody;
    if ((nodeMode.value === "selected" && !hasExtended) || (nodeMode.value === "full" && !hasWholebody)) {
      nodeMode.value = "hidden";
    }

    let frame = 0;
    let timer = null;
    let yaw = -0.86;
    let pitch = -0.32;
    let zoom = 1.08;
    let skeletonScale = initialSkeletonScale;
    let isDragging = false;
    let lastPointer = { x: 0, y: 0 };

    const regionViewer = Human17RegionModel.createHuman17RegionViewer(canvas, {
      radius,
      pairs,
      boneColors,
      yaw,
      pitch,
      zoom,
      skeletonScale,
    });
    if (hasExtended) regionViewer.setExtendedSequence(extendedPoseData);
    else regionViewer.setSequence(poseData);

    function padFrame(value) {
      return String(value).padStart(4, "0");
    }

    function resizeCanvas() {
      regionViewer.resize();
    }

    function drawScene() {
      regionViewer.setView({ yaw, pitch, zoom, skeletonScale }, false);
      regionViewer.setOptions({
        regionsVisible: regionToggle.checked,
        skeletonVisible: skeletonToggle.checked,
        thickness: Number(regionThickness.value),
        extendedVisible: !hasExtended || extendedToggle.checked,
        feetVisible: !hasExtended || feetToggle.checked,
        handsVisible: !hasExtended || handsToggle.checked,
        faceMarkersVisible: !hasExtended || faceToggle.checked,
        nodeMode: nodeMode.value,
        nodeThreshold: Number(nodeThreshold.value),
      }, false);
      if (hasExtended) regionViewer.setExtendedPose(extendedPoseData[frame], extendedValid[frame], false);
      else regionViewer.setPose(poseData[frame], false);
      const visibleNodes = regionViewer.setNodeOverlayData({
        selectedPose: hasExtended ? extendedPoseData[frame] : null,
        selectedValid: hasExtended ? extendedValid[frame] : null,
        selectedScores: hasExtended ? extendedScores[frame] : null,
        wholebodyPose: hasWholebody ? wholebodyPoseData[frame] : null,
        wholebodyValid: hasWholebody ? wholebodyValid[frame] : null,
        wholebodyScores: hasWholebody ? wholebodyScores[frame] : null,
      });
      nodeCount.textContent = `RTMW3D nodes: ${visibleNodes}`;
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
    regionToggle.addEventListener("change", drawScene);
    skeletonToggle.addEventListener("change", drawScene);
    regionThickness.addEventListener("input", drawScene);
    extendedToggle.addEventListener("change", drawScene);
    feetToggle.addEventListener("change", drawScene);
    handsToggle.addEventListener("change", drawScene);
    faceToggle.addEventListener("change", drawScene);
    nodeMode.addEventListener("change", drawScene);
    nodeThreshold.addEventListener("input", drawScene);

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
    parser.add_argument(
        "--extended-node-mode",
        choices=("hidden", "selected", "full"),
        default="hidden",
        help="Initial RTMW3D diagnostic-node display mode.",
    )
    parser.add_argument(
        "--node-confidence-threshold",
        type=float,
        default=0.25,
        help="Initial confidence threshold for RTMW3D diagnostic nodes.",
    )
    args = parser.parse_args()

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input_3d_npz, allow_pickle=True)
    pred3d = data["pred3d_root_relative_image_units"].astype(np.float32)
    display = np.stack([to_display_coords(pose) for pose in pred3d], axis=0)
    extended_display = None
    extended_valid = None
    extended_scores = None
    wholebody_display = None
    wholebody_valid = None
    wholebody_scores = None
    if "extended_pose_3d" in data.files:
        extended = np.asarray(data["extended_pose_3d"], dtype=np.float32)
        if extended.ndim == 3 and extended.shape[1] >= 17:
            extended_display = np.stack([to_display_coords(pose) for pose in extended], axis=0)
            if "extended_valid" in data.files:
                extended_valid = np.asarray(data["extended_valid"], dtype=bool)
            else:
                extended_valid = np.all(np.isfinite(extended_display), axis=-1)
            if "extended_scores" in data.files:
                extended_scores = np.asarray(data["extended_scores"], dtype=np.float32)
            else:
                extended_scores = extended_valid.astype(np.float32)
            frame_count = min(len(display), len(extended_display))
            display = display[:frame_count]
            extended_display = extended_display[:frame_count]
            extended_valid = extended_valid[:frame_count]
            extended_scores = extended_scores[:frame_count]
            display = extended_display[:, :17]
    if "wholebody_pose_3d_aligned" in data.files:
        wholebody = np.asarray(data["wholebody_pose_3d_aligned"], dtype=np.float32)
        if wholebody.ndim == 3 and wholebody.shape[1] == 133:
            wholebody_display = np.stack([to_display_coords(pose) for pose in wholebody], axis=0)
            if "wholebody_valid" in data.files:
                wholebody_valid = np.asarray(data["wholebody_valid"], dtype=bool)
            else:
                wholebody_valid = np.all(np.isfinite(wholebody_display), axis=-1)
            if "wholebody_scores" in data.files:
                wholebody_scores = np.asarray(data["wholebody_scores"], dtype=np.float32)
            else:
                wholebody_scores = wholebody_valid.astype(np.float32)
            frame_count = min(len(display), len(wholebody_display))
            display = display[:frame_count]
            wholebody_display = wholebody_display[:frame_count]
            wholebody_valid = wholebody_valid[:frame_count]
            wholebody_scores = wholebody_scores[:frame_count]
            if extended_display is not None:
                extended_display = extended_display[:frame_count]
                extended_valid = extended_valid[:frame_count]
                extended_scores = extended_scores[:frame_count]
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
            if extended_display is not None:
                extended_display = extended_display[:extracted]
                extended_valid = extended_valid[:extracted]
                extended_scores = extended_scores[:extracted]
            if wholebody_display is not None:
                wholebody_display = wholebody_display[:extracted]
                wholebody_valid = wholebody_valid[:extracted]
                wholebody_scores = wholebody_scores[:extracted]
    elif args.left_frame_dir:
        left_dir = Path(args.left_frame_dir)
        frame_count = count_frame_files(left_dir)
        if frame_count == 0:
            raise SystemExit(f"No JPG frames found in: {left_dir}")
        if frame_count < len(display):
            print(f"warning: left frames={frame_count}, but 3D data has {len(display)} frames")
            display = display[:frame_count]
            if extended_display is not None:
                extended_display = extended_display[:frame_count]
                extended_valid = extended_valid[:frame_count]
                extended_scores = extended_scores[:frame_count]
            if wholebody_display is not None:
                wholebody_display = wholebody_display[:frame_count]
                wholebody_valid = wholebody_valid[:frame_count]
                wholebody_scores = wholebody_scores[:frame_count]
    else:
        raise SystemExit("Either --left-video or --left-frame-dir is required")

    html_text = HTML_TEMPLATE
    html_text = html_text.replace("__TITLE__", html.escape(args.title))
    html_text = html_text.replace("__REGION_MODEL_BUNDLE__", read_region_model_bundle())
    html_text = html_text.replace("__POSE_DATA__", json.dumps(compact_pose_data(display), separators=(",", ":")))
    if extended_display is None:
        html_text = html_text.replace("__HAS_EXTENDED__", "false")
        html_text = html_text.replace("__EXTENDED_POSE_DATA__", "[]")
        html_text = html_text.replace("__EXTENDED_VALID__", "[]")
        html_text = html_text.replace("__EXTENDED_SCORES__", "[]")
    else:
        pose_json, valid_json = compact_extended_payload(extended_display, extended_valid)
        html_text = html_text.replace("__HAS_EXTENDED__", "true")
        html_text = html_text.replace("__EXTENDED_POSE_DATA__", json.dumps(pose_json, separators=(",", ":")))
        html_text = html_text.replace("__EXTENDED_VALID__", json.dumps(valid_json, separators=(",", ":")))
        html_text = html_text.replace(
            "__EXTENDED_SCORES__",
            json.dumps(compact_scores(extended_scores), separators=(",", ":")),
        )
    if wholebody_display is None:
        html_text = html_text.replace("__HAS_WHOLEBODY__", "false")
        html_text = html_text.replace("__WHOLEBODY_POSE_DATA__", "[]")
        html_text = html_text.replace("__WHOLEBODY_VALID__", "[]")
        html_text = html_text.replace("__WHOLEBODY_SCORES__", "[]")
    else:
        whole_pose_json, whole_valid_json = compact_extended_payload(
            wholebody_display, wholebody_valid
        )
        html_text = html_text.replace("__HAS_WHOLEBODY__", "true")
        html_text = html_text.replace(
            "__WHOLEBODY_POSE_DATA__",
            json.dumps(whole_pose_json, separators=(",", ":")),
        )
        html_text = html_text.replace(
            "__WHOLEBODY_VALID__",
            json.dumps(whole_valid_json, separators=(",", ":")),
        )
        html_text = html_text.replace(
            "__WHOLEBODY_SCORES__",
            json.dumps(compact_scores(wholebody_scores), separators=(",", ":")),
        )
    node_mode = args.extended_node_mode
    if node_mode == "selected" and extended_display is None:
        print("warning: selected RTMW3D nodes requested but extended_pose_3d is unavailable; using hidden")
        node_mode = "hidden"
    if node_mode == "full" and wholebody_display is None:
        print("warning: full RTMW3D nodes requested but aligned WholeBody-133 data is unavailable; using hidden")
        node_mode = "hidden"
    html_text = html_text.replace("__NODE_MODE__", node_mode)
    node_threshold = max(0.0, min(1.0, float(args.node_confidence_threshold)))
    html_text = html_text.replace("__NODE_THRESHOLD__", f"{node_threshold:.2f}")
    html_text = html_text.replace("__PAIRS__", json.dumps(H36M_PAIRS, separators=(",", ":")))
    html_text = html_text.replace("__BONE_COLORS__", json.dumps(build_bone_colors(), separators=(",", ":")))
    html_text = html_text.replace("__LEFT_DIR__", html_relative_dir(left_dir, out_html.parent))
    html_text = html_text.replace("__FPS__", f"{args.fps:.8f}")
    html_text = html_text.replace("__RADIUS__", f"{radius:.6f}")
    html_text = html_text.replace("__SKELETON_SCALE__", f"{skeleton_scale:.6f}")
    html_text = html_text.replace("__LAST_FRAME__", str(len(display) - 1))

    out_html.write_text(html_text, encoding="utf-8")
    html_size_mb = len(html_text.encode("utf-8")) / (1024 * 1024)
    print(f"frames: {len(display)}")
    print(f"radius: {radius:.3f}")
    print(f"skeleton scale: {skeleton_scale:.3f}")
    print(f"RTMW3D node mode: {node_mode}")
    print(f"HTML size: {html_size_mb:.1f} MB")
    if html_size_mb > 100:
        print("warning: full WholeBody data produced a large self-contained HTML; selected mode is recommended for long videos")
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
