import argparse
import html
import json
import os
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults if you prefer running this script without command args.
DEFAULT_VIDEO_NAME = "test1"
DEFAULT_VIDEO_DIR = ROOT / "input_videos" / "finefs_test"
DEFAULT_H36M_JSON = ROOT / "outputs" / "processed_2d" / f"{DEFAULT_VIDEO_NAME}_h36m.json"
DEFAULT_H36M_NPZ = ROOT / "outputs" / "processed_2d" / f"{DEFAULT_VIDEO_NAME}_h36m.npz"
DEFAULT_OUT_HTML = ROOT / "outputs" / "pose_editor" / f"{DEFAULT_VIDEO_NAME}_2d_pose_editor.html"
DEFAULT_TITLE = f"{DEFAULT_VIDEO_NAME} 2D Pose Editor"


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


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #171a1d;
      --panel2: #1e2328;
      --line: #31373d;
      --text: #edf2f7;
      --muted: #9aa5af;
      --blue: #3b82f6;
      --green: #16a34a;
      --red: #dc2626;
      --yellow: #f59e0b;
      --pink: #ec4899;
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
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #15181b;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 12px;
      min-height: 0;
      padding: 12px;
    }
    .stage {
      display: grid;
      place-items: center;
      min-width: 0;
      min-height: 0;
      border: 1px solid var(--line);
      background: #0b0d0f;
      overflow: hidden;
    }
    .video-wrap {
      position: relative;
      width: min(100%, 1280px);
      background: #050607;
    }
    video {
      display: block;
      width: 100%;
      height: auto;
    }
    canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
      touch-action: none;
    }
    canvas.dragging { cursor: grabbing; }
    aside {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      gap: 10px;
      min-height: 0;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
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
      background: var(--panel2);
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover, select:hover, input[type="number"]:hover { border-color: var(--blue); }
    button.primary { border-color: #2563eb; background: #1d4ed8; }
    button.warn { border-color: #a16207; background: #713f12; }
    button:disabled { opacity: 0.45; cursor: default; }
    select { width: 100%; }
    input[type="range"] { width: 100%; accent-color: var(--blue); }
    input[type="number"] { width: 84px; }
    .joint-list {
      overflow: auto;
      border: 1px solid var(--line);
      background: #111417;
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
    .joint.low {
      color: #fde68a;
      background: #2a2112;
    }
    .joint.low.active {
      color: #fff7c2;
      background: #3a2d16;
    }
    .joint .dot {
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: var(--blue);
    }
    .joint.low .dot { background: var(--yellow); }
    .joint.edited .dot { background: var(--pink); }
    .score {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .joint.low .score { color: #fde68a; }
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
    .frame-label {
      min-width: 116px;
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; }
      aside { grid-template-rows: auto auto 320px auto; }
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
      <section class="stage">
        <div class="video-wrap" id="videoWrap">
          <video id="video" muted playsinline preload="metadata"></video>
          <canvas id="overlay"></canvas>
        </div>
      </section>

      <aside>
        <section class="panel">
          <div class="grid">
            <button id="saveBtn" class="primary" type="button">Save JSON</button>
            <button id="downloadBtn" type="button">Download</button>
            <button id="undoBtn" type="button">Undo</button>
            <button id="resetFrameBtn" class="warn" type="button">Reset Frame</button>
          </div>
        </section>

        <section class="panel">
          <div class="row">
            <select id="jointSelect"></select>
          </div>
          <div class="grid" style="margin-top:8px;">
            <button id="interpJointBtn" type="button">Interp Joint</button>
            <button id="interpAllBtn" type="button">Interp All</button>
            <button id="prevEditedBtn" type="button">Prev Edit</button>
            <button id="nextEditedBtn" type="button">Next Edit</button>
          </div>
          <div class="row" style="margin-top:8px;">
            <input id="scoreThreshold" type="number" min="0" max="1" step="0.05" value="0.25" />
            <button id="nextLowBtn" type="button">Next Low</button>
          </div>
        </section>

        <section class="joint-list" id="jointList"></section>

        <section class="panel small">
          Drag a joint to correct the current frame. Click empty space to move the selected joint. Use edited keyframes and interpolation for long clips.
        </section>
      </aside>
    </main>

    <div class="timeline">
      <button id="playPause" type="button">Play</button>
      <button id="prevFrame" type="button">Prev</button>
      <button id="nextFrame" type="button">Next</button>
      <input id="timeline" type="range" min="0" max="0" step="1" value="0" />
      <select id="speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="1.5">1.5x</option>
        <option value="2">2x</option>
      </select>
      <div class="frame-label" id="frameLabel">0 / 0</div>
    </div>
  </div>

  <script>
    const payload = __PAYLOAD__;
    const videoSrc = __VIDEO_SRC__;
    const jointNames = payload.joint_names;
    const pairs = payload.pairs;
    const frames = payload.frames;
    const original = JSON.parse(JSON.stringify(frames.map((f) => f.keypoints)));
    const fps = Number(payload.fps || 30);
    const imageWidth = Number(payload.image_width);
    const imageHeight = Number(payload.image_height);
    const lastFrame = frames.length - 1;

    const video = document.getElementById("video");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    const timeline = document.getElementById("timeline");
    const frameLabel = document.getElementById("frameLabel");
    const timeLabel = document.getElementById("timeLabel");
    const stateLabel = document.getElementById("stateLabel");
    const editedLabel = document.getElementById("editedLabel");
    const jointList = document.getElementById("jointList");
    const jointSelect = document.getElementById("jointSelect");
    const playPause = document.getElementById("playPause");
    const undoBtn = document.getElementById("undoBtn");
    const speed = document.getElementById("speed");
    const scoreThresholdInput = document.getElementById("scoreThreshold");

    video.src = videoSrc;
    timeline.max = String(lastFrame);

    let frame = 0;
    let selectedJoint = 0;
    let draggingJoint = null;
    let dragSnapshot = null;
    let timer = null;
    let videoFrameRequest = null;
    let history = [];
    const edited = frames.map(() => Array(jointNames.length).fill(false));

    function boneColor(a, b) {
      const left = ["0-4", "4-5", "5-6", "8-11", "11-12", "12-13"];
      const right = ["0-1", "1-2", "2-3", "8-14", "14-15", "15-16"];
      const key = `${a}-${b}`;
      if (left.includes(key)) return "#16a34a";
      if (right.includes(key)) return "#dc2626";
      return "#2563eb";
    }

    function resizeCanvas() {
      const rect = video.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function scaleXY() {
      const rect = canvas.getBoundingClientRect();
      return { sx: rect.width / imageWidth, sy: rect.height / imageHeight, rect };
    }

    function toCanvas(point) {
      const { sx, sy } = scaleXY();
      return { x: point[0] * sx, y: point[1] * sy };
    }

    function fromCanvas(x, y) {
      const { sx, sy } = scaleXY();
      return {
        x: Math.max(0, Math.min(imageWidth, x / sx)),
        y: Math.max(0, Math.min(imageHeight, y / sy)),
      };
    }

    function currentPose() {
      return frames[frame].keypoints;
    }

    function lowConfidenceThreshold() {
      const value = Number(scoreThresholdInput.value);
      return Number.isFinite(value) ? value : 0.25;
    }

    function draw() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const pose = currentPose();
      const threshold = lowConfidenceThreshold();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (const [a, b] of pairs) {
        const pa = toCanvas(pose[a]);
        const pb = toCanvas(pose[b]);
        ctx.strokeStyle = boneColor(a, b);
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      for (let i = 0; i < pose.length; i += 1) {
        const p = toCanvas(pose[i]);
        const score = Number(pose[i][2] || 0);
        ctx.beginPath();
        ctx.fillStyle = edited[frame][i] ? "#ec4899" : score < threshold ? "#f59e0b" : "#111827";
        ctx.strokeStyle = i === selectedJoint ? "#fef08a" : "#ffffff";
        ctx.lineWidth = i === selectedJoint ? 3 : 1.5;
        ctx.arc(p.x, p.y, i === selectedJoint ? 7 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    function seekVideo() {
      const t = Math.max(0, frame / fps);
      // At 60 FPS a frame is only 16.7 ms.  A fixed 35 ms tolerance
      // skips two adjacent frames and makes the skeleton visibly lag.
      const tolerance = 1 / Math.max(fps * 4, 1);
      if (Math.abs(video.currentTime - t) > tolerance) {
        video.currentTime = t;
      }
    }

    function updateJointPanel() {
      const pose = currentPose();
      const threshold = lowConfidenceThreshold();
      for (const el of jointList.children) {
        const j = Number(el.dataset.joint);
        const score = Number(pose[j][2] || 0);
        el.classList.toggle("active", j === selectedJoint);
        el.classList.toggle("edited", edited[frame][j]);
        el.classList.toggle("low", score < threshold);
        el.title = score < threshold ? `low confidence: ${score.toFixed(3)} < ${threshold}` : "";
        el.querySelector(".score").textContent = score.toFixed(2);
      }
      jointSelect.value = String(selectedJoint);
    }

    function updateLabels() {
      timeline.value = String(frame);
      frameLabel.textContent = `${frame} / ${lastFrame}`;
      timeLabel.textContent = `${(frame / fps).toFixed(3)}s`;
      const count = edited.reduce((sum, row) => sum + row.filter(Boolean).length, 0);
      editedLabel.textContent = `${count} edits`;
      undoBtn.disabled = history.length === 0;
    }

    function setFrame(value) {
      frame = Math.max(0, Math.min(lastFrame, Math.round(value)));
      seekVideo();
      updateJointPanel();
      updateLabels();
      draw();
    }

    function pause() {
      if (typeof timer === "number") clearInterval(timer);
      if (videoFrameRequest !== null && video.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(videoFrameRequest);
      }
      videoFrameRequest = null;
      video.pause();
      timer = null;
      playPause.textContent = "Play";
      stateLabel.textContent = "paused";
    }

    function syncFrameFromVideo(mediaTime) {
      const nextFrame = Math.max(0, Math.min(lastFrame, Math.round(mediaTime * fps)));
      if (nextFrame === frame) return;
      frame = nextFrame;
      updateJointPanel();
      updateLabels();
      draw();
    }

    function onVideoFrame(_, metadata) {
      if (!timer) return;
      syncFrameFromVideo(metadata.mediaTime);
      if (frame >= lastFrame || video.ended) {
        pause();
        return;
      }
      videoFrameRequest = video.requestVideoFrameCallback(onVideoFrame);
    }

    function play() {
      if (timer) return;
      if (frame >= lastFrame) setFrame(0);
      playPause.textContent = "Pause";
      stateLabel.textContent = "playing";
      timer = true;
      video.playbackRate = Number(speed.value);
      video.play().then(() => {
        if (!timer) return;
        if (video.requestVideoFrameCallback) {
          videoFrameRequest = video.requestVideoFrameCallback(onVideoFrame);
          return;
        }
        // Fallback for older browsers: read the video's actual playhead
        // instead of seeking it once per skeleton frame.
        timer = setInterval(() => {
          syncFrameFromVideo(video.currentTime);
          if (frame >= lastFrame || video.ended) pause();
        }, 16);
      }).catch(() => pause());
    }

    function nearestJoint(x, y) {
      let best = null;
      let bestDist = Infinity;
      const pose = currentPose();
      for (let i = 0; i < pose.length; i += 1) {
        const p = toCanvas(pose[i]);
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      return bestDist <= 18 ? best : null;
    }

    function markEdited(joint) {
      edited[frame][joint] = true;
      frames[frame].keypoints[joint][2] = Math.max(Number(frames[frame].keypoints[joint][2] || 0), 1.0);
      updateJointPanel();
      updateLabels();
    }

    function moveJoint(joint, x, y) {
      const p = fromCanvas(x, y);
      frames[frame].keypoints[joint][0] = p.x;
      frames[frame].keypoints[joint][1] = p.y;
      markEdited(joint);
      draw();
    }

    function pushHistory(action) {
      history.push(action);
      if (history.length > 200) history.shift();
      updateLabels();
    }

    function restorePoint(frameIndex, joint, point, wasEdited) {
      frames[frameIndex].keypoints[joint] = [...point];
      edited[frameIndex][joint] = Boolean(wasEdited);
    }

    function interpolateJoint(joint) {
      const keys = [];
      for (let i = 0; i <= lastFrame; i += 1) {
        if (edited[i][joint]) keys.push(i);
      }
      if (keys.length < 2) return;
      const before = frames.map((f) => [...f.keypoints[joint]]);
      const editedBefore = edited.map((row) => row[joint]);
      for (let k = 0; k < keys.length - 1; k += 1) {
        const a = keys[k];
        const b = keys[k + 1];
        const pa = frames[a].keypoints[joint];
        const pb = frames[b].keypoints[joint];
        for (let i = a + 1; i < b; i += 1) {
          const t = (i - a) / (b - a);
          frames[i].keypoints[joint] = [
            pa[0] * (1 - t) + pb[0] * t,
            pa[1] * (1 - t) + pb[1] * t,
            pa[2] * (1 - t) + pb[2] * t,
          ];
          edited[i][joint] = true;
        }
      }
      pushHistory({ type: "joint_series", joint, before, editedBefore });
      setFrame(frame);
    }

    function editedFrameIndices() {
      const out = [];
      for (let i = 0; i <= lastFrame; i += 1) {
        if (edited[i].some(Boolean)) out.push(i);
      }
      return out;
    }

    function findNextLowConfidence(threshold) {
      const joints = jointNames.length;
      const start = frame * joints + selectedJoint + 1;
      for (let flat = start; flat < frames.length * joints; flat += 1) {
        const nextFrame = Math.floor(flat / joints);
        const nextJoint = flat % joints;
        const score = Number(frames[nextFrame].keypoints[nextJoint][2] || 0);
        if (score < threshold) {
          return { frame: nextFrame, joint: nextJoint, score };
        }
      }
      return null;
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
              keypoint: frames[i].keypoints[j],
            });
          }
        }
      }
      return {
        source_json: payload.source_json,
        source_npz: payload.source_npz,
        source_video: payload.source_video,
        correction_format: "h36m_17_manual_2d_v1",
        format: "h36m_17",
        image_width: imageWidth,
        image_height: imageHeight,
        fps,
        joint_names: jointNames,
        pairs,
        manual_edits: manualEdits,
        frames,
      };
    }

    async function saveJson(preferPicker) {
      const text = JSON.stringify(makeExportPayload(), null, 2);
      const suggestedName = `${payload.output_stem}_corrected_h36m.json`;
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
        item.innerHTML = `<span class="dot"></span><span>${i} ${jointNames[i]}</span><span class="score">0.00</span>`;
        item.addEventListener("click", () => {
          selectedJoint = i;
          updateJointPanel();
          draw();
        });
        jointList.appendChild(item);
      }
    }

    canvas.addEventListener("pointerdown", (event) => {
      pause();
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = nearestJoint(x, y);
      draggingJoint = hit === null ? selectedJoint : hit;
      selectedJoint = draggingJoint;
      dragSnapshot = {
        frame,
        joint: draggingJoint,
        before: [...frames[frame].keypoints[draggingJoint]],
        editedBefore: edited[frame][draggingJoint],
      };
      moveJoint(draggingJoint, x, y);
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      if (draggingJoint === null) return;
      const rect = canvas.getBoundingClientRect();
      moveJoint(draggingJoint, event.clientX - rect.left, event.clientY - rect.top);
    });

    canvas.addEventListener("pointerup", (event) => {
      if (dragSnapshot) {
        pushHistory(dragSnapshot);
      }
      draggingJoint = null;
      dragSnapshot = null;
      canvas.classList.remove("dragging");
      canvas.releasePointerCapture(event.pointerId);
    });

    document.getElementById("playPause").addEventListener("click", () => timer ? pause() : play());
    document.getElementById("prevFrame").addEventListener("click", () => { pause(); setFrame(frame - 1); });
    document.getElementById("nextFrame").addEventListener("click", () => { pause(); setFrame(frame + 1); });
    timeline.addEventListener("input", () => { pause(); setFrame(Number(timeline.value)); });
    speed.addEventListener("change", () => { if (timer) { pause(); play(); } });
    jointSelect.addEventListener("change", () => {
      selectedJoint = Number(jointSelect.value);
      updateJointPanel();
      draw();
    });
    scoreThresholdInput.addEventListener("change", () => {
      updateJointPanel();
      draw();
    });

    document.getElementById("saveBtn").addEventListener("click", () => saveJson(true).catch((err) => alert(err.message)));
    document.getElementById("downloadBtn").addEventListener("click", () => saveJson(false));
    undoBtn.addEventListener("click", () => {
      const item = history.pop();
      if (!item) return;
      if (item.type === "joint_series") {
        for (let i = 0; i <= lastFrame; i += 1) {
          frames[i].keypoints[item.joint] = [...item.before[i]];
          edited[i][item.joint] = item.editedBefore[i];
        }
      } else if (item.type === "frame") {
        frames[item.frame].keypoints = JSON.parse(JSON.stringify(item.before));
        edited[item.frame] = [...item.editedBefore];
      } else {
        restorePoint(item.frame, item.joint, item.before, item.editedBefore);
      }
      setFrame(frame);
    });
    document.getElementById("resetFrameBtn").addEventListener("click", () => {
      const before = JSON.parse(JSON.stringify(frames[frame].keypoints));
      const editedBefore = [...edited[frame]];
      frames[frame].keypoints = JSON.parse(JSON.stringify(original[frame]));
      edited[frame].fill(false);
      pushHistory({ type: "frame", frame, before, editedBefore });
      setFrame(frame);
    });
    document.getElementById("interpJointBtn").addEventListener("click", () => interpolateJoint(selectedJoint));
    document.getElementById("interpAllBtn").addEventListener("click", () => {
      for (let j = 0; j < jointNames.length; j += 1) interpolateJoint(j);
    });
    document.getElementById("prevEditedBtn").addEventListener("click", () => {
      const keys = editedFrameIndices().filter((i) => i < frame);
      if (keys.length) setFrame(keys[keys.length - 1]);
    });
    document.getElementById("nextEditedBtn").addEventListener("click", () => {
      const keys = editedFrameIndices().filter((i) => i > frame);
      if (keys.length) setFrame(keys[0]);
    });
    document.getElementById("nextLowBtn").addEventListener("click", () => {
      const thr = lowConfidenceThreshold();
      const found = findNextLowConfidence(thr);
      if (found) {
        selectedJoint = found.joint;
        setFrame(found.frame);
        const item = jointList.querySelector(`[data-joint="${found.joint}"]`);
        if (item) item.scrollIntoView({ block: "nearest" });
        return;
      }
      alert(`No more joints below ${thr}.`);
    });

    window.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        timer ? pause() : play();
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
    window.addEventListener("resize", resizeCanvas);
    video.addEventListener("loadedmetadata", () => {
      resizeCanvas();
      seekVideo();
    });
    video.addEventListener("seeked", draw);

    buildJointControls();
    setFrame(0);
    requestAnimationFrame(resizeCanvas);
  </script>
</body>
</html>
"""


def html_relative_path(path, base_dir):
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve()
    try:
        rel_path = os.path.relpath(path, base_dir)
        return rel_path.replace(os.sep, "/")
    except ValueError:
        # Windows cannot calculate relative paths across drive letters.
        # A file URI lets an HTML viewer on another drive load the source video.
        return path.as_uri()


def video_meta(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, frames


def load_h36m_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise SystemExit(f"No frames found in: {path}")
    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) != 17:
            raise SystemExit(f"Invalid H36M keypoints at frame index {idx}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h36m-json",
        default=str(DEFAULT_H36M_JSON),
        help="H36M JSON path. Default is derived from DEFAULT_VIDEO_NAME.",
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO_DIR / f"{DEFAULT_VIDEO_NAME}.mp4"),
        help="Input video path. Default is derived from DEFAULT_VIDEO_NAME.",
    )
    parser.add_argument(
        "--out-html",
        default=str(DEFAULT_OUT_HTML),
        help="Output editor HTML path. Default is derived from DEFAULT_VIDEO_NAME.",
    )
    parser.add_argument(
        "--source-npz",
        default=str(DEFAULT_H36M_NPZ),
        help="Source H36M NPZ path. Default is derived from DEFAULT_VIDEO_NAME.",
    )
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()

    h36m_json = Path(args.h36m_json)
    video_path = Path(args.video)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    data = load_h36m_json(h36m_json)
    width, height, video_fps, video_frames = video_meta(video_path)
    fps = args.fps or video_fps
    if video_frames and abs(video_frames - len(data["frames"])) > 2:
        print(f"warning: video frames={video_frames}, pose frames={len(data['frames'])}")

    payload = {
        "source_json": str(h36m_json),
        "source_npz": str(Path(args.source_npz)) if args.source_npz else "",
        "source_video": str(video_path),
        "output_stem": out_html.stem,
        "format": "h36m_17",
        "image_width": data.get("image_width", width),
        "image_height": data.get("image_height", height),
        "fps": fps,
        "joint_names": data.get("joint_names", H36M_NAMES),
        "pairs": H36M_PAIRS,
        "frames": data["frames"],
    }

    html_text = HTML_TEMPLATE
    html_text = html_text.replace("__TITLE__", html.escape(args.title))
    html_text = html_text.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    html_text = html_text.replace("__VIDEO_SRC__", json.dumps(html_relative_path(video_path, out_html.parent)))

    out_html.write_text(html_text, encoding="utf-8")
    print(f"frames: {len(data['frames'])}")
    print(f"video: {width}x{height} @ {fps:.3f} fps")
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
