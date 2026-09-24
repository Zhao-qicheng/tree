# Human17 Region Model 交接说明

本文面向下一位维护者，覆盖 `region_model/README.md` 中提到的所有仓库内路径、上下游调用链、数据协议、运行方法、验证方式和已知风险。除非特别说明，命令都从仓库根目录执行。

## 1. 一句话理解

这个模块把 MotionAGFormer 输出的 Human3.6M 17 点三维姿态画成浏览器中的轻量人体区域模型；可选地把 RTMW3D 的 COCO-WholeBody 133 点对齐到核心骨架，筛出脚、手指尖和面部方向点，形成 39 点姿态。

它是查看器，不是物理模型，不提供碰撞、动力学、MuJoCo 或 PyBullet 能力。

## 2. 路径标记

文档使用以下标记：

- **[入口]**：开发者或流水线直接执行的文件。
- **[核心]**：主要算法或渲染实现。
- **[协议]**：跨 Python/JavaScript 共用的数据约定。
- **[配置]**：可调整默认参数。
- **[测试]**：自动验证入口。
- **[生成物]**：由命令生成，通常不应手工修改。
- **[外部]**：不在本仓库中，需要另行准备。
- **[本地数据]**：被 `.gitignore` 忽略，不应提交。

## 3. 推荐阅读顺序

1. **[说明] `region_model/README.md`**：用途、最短构建命令、浏览器 API 和 39 点概览。
2. **[入口] `scripts/run_video_pose_pipeline.py`**：从视频到交互 HTML 的完整调用链。
3. **[协议] `utils/extended_pose_schema.py`**：17/39/133 点编号与映射的唯一 Python 定义。
4. **[核心] `utils/extended_pose_fusion.py`**：133 点到 39 点的对齐、过滤、插值和平滑。
5. **[核心] `region_model/src/profile.js`**：JavaScript 侧同一套点位协议、体段尺寸和方向计算。
6. **[核心] `region_model/src/viewer.js`**：Three.js 场景、人体几何、节点叠加和公开 API。
7. **[入口] `scripts/build_interactive_3d_viewer.py`**：把姿态、图片和 JS 包装进最终 HTML。
8. **[测试] `tests/test_extended_pose.py`、`tests/test_interactive_3d_viewer.py`**：协议和集成行为的可执行说明。

## 4. 总体调用链

```text
输入视频
  -> MMPose 2D 人体检测/姿态
  -> select_main_skater_h36m.py
  -> H36M-17 二维姿态
  -> [可选] refine_2d_h36m.py
  -> lift_2d_to_3d_motionagformer_ap3d.py
  -> H36M-17 三维姿态
  -> [可选] refine_3d_pose.py
  -> [可选] infer_rtmw3d_wholebody.py -> WholeBody-133
  -> [可选] fuse_extended_pose.py -> H36M-17 + 扩展 22 点 = 39 点
  -> build_interactive_3d_viewer.py
       + region_model/dist/human17-region-model.js
  -> 自包含交互 HTML
```

扩展分支是可选分支。RTMW3D 推理或融合失败时，总流水线会保留 17 点结果，并继续生成只含核心骨架的查看器。

## 5. `region_model/` 内全部路径

### [说明] `region_model/README.md`

面向使用者的短说明，包含构建、浏览器 API、39 点顺序和扩展流水线示例。新增或变更公开 API、点位协议、构建产物名称时，应同步更新它和本文。

### [入口] `region_model/package.json`

- 包名：`human17-region-model`，当前版本 `0.1.0`。
- 模块类型：ES Module。
- 运行依赖：`three@0.180.0`。
- 开发依赖：`esbuild@0.25.10`。
- `npm run build`：以 `src/index.js` 为入口，生成 IIFE 全局包。
- 浏览器全局名：`Human17RegionModel`。
- 输出：`dist/human17-region-model.js`，Three.js 已打包，无 CDN 依赖。
- `npm test` 当前指向 `test/*.test.mjs`，但仓库中没有 `region_model/test/`；在补齐 Node 测试前，该命令不是有效验证入口。

### [生成物] `region_model/package-lock.json`

锁定 npm 依赖版本。依赖变更后用 npm 重新生成，不要手工编辑。

### [配置] `region_model/.gitignore`

忽略 `node_modules/` 和 `.npm-cache/`。`dist/` 没有被忽略，因此构建包属于仓库内容。

### [入口] `region_model/src/index.js`

公共导出汇总。它导出：

- 查看器：`Human17RegionViewer`、`createHuman17RegionViewer`。
- 点位/骨骼常量：`HUMAN17_*`、`EXTENDED_*`、`SELECTED_EXTRA_*`、`WHOLEBODY133_BONES`。
- 校验和转换：`isFinitePoint`、`isHuman17Pose`、`isExtendedPose`、`corePoseFrom`、`posePoint`。
- 几何与统计：`makeFrame`、`computeTorsoFrame`、`computeHeadFrame`、`computeFootFrame`、`computeHandFrame`、`pointDistance`、`robustMedian`、`estimateHuman17Profile`。

增加公共函数时，需要在这里显式导出，然后重新构建 `dist`。

### [核心][协议] `region_model/src/profile.js`

负责不依赖 Three.js 的数据和几何逻辑：

- 定义 Human17 的 17 个关节、16 条骨骼及默认颜色。
- 定义 39 点名称、父节点和选中扩展节点。
- 定义 WholeBody-133 的诊断骨骼连线。
- 校验点、17 点姿态和 39 点姿态。
- 从肩、躯干、眼耳鼻、脚趾脚跟和指尖估计躯干、头、脚、手的局部坐标系。
- 从最多 360 个均匀采样帧估计稳定体段长度和半径。
- 使用中位数与 MAD 剔除异常长度，避免人体粗细随帧抖动。

Python 与 JavaScript 各维护了一份点位常量。修改 `utils/extended_pose_schema.py` 时，必须同步检查本文件。

### [核心] `region_model/src/viewer.js`

Three.js 查看器实现，主要组成如下：

- 正交相机、参考立方体、灯光和 WebGL 渲染器。
- 胶囊四肢/骨盆/肩颈、锥台躯干、椭球头部。
- 17 点骨架线与关节球。
- 扩展方向几何：面罩、胸部方向、定向脚、手掌和手指。
- `selected` 39 点与 `full` 133 点诊断叠加层。
- 视角、缩放、人体缩放、粗细、显示开关、置信度阈值。
- `dispose()` 释放几何、材质和 WebGL 渲染器。

构造参数及默认行为：

- `radius=1`，决定参考空间和相机范围。
- `yaw=0`、`pitch=0`、`zoom=1`、`skeletonScale=1`。
- `regionsVisible=true`、`skeletonVisible=true`。
- `thickness=1`，后续设置会限制在 `0.65–1.55`。
- 扩展几何默认可见，但只有传入有效 39 点时才会出现。
- `nodeMode="hidden"`，可取 `hidden | selected | full`。
- `nodeThreshold=0.25`，后续设置会限制在 `0–1`。

### [生成物] `region_model/dist/human17-region-model.js`

由 esbuild 生成的压缩 IIFE，包含 Three.js。`build_interactive_3d_viewer.py` 会读取并内嵌它，所以源代码改变但未重建时，最终 HTML 仍会使用旧逻辑。

不要直接编辑此文件。正确流程是修改 `src/`，再运行：

```powershell
Set-Location .\region_model
npm install
npm run build
Set-Location ..
```

提交前同时检查 `src/` 与 `dist/` 的改动。

## 6. 浏览器公开 API

最小用法：

```html
<canvas id="scene"></canvas>
<script src="region_model/dist/human17-region-model.js"></script>
<script>
  const viewer = Human17RegionModel.createHuman17RegionViewer(
    document.getElementById("scene"),
    { radius: 2.2, zoom: 1.08, yaw: -0.86, pitch: -0.32 }
  );
  viewer.setSequence(allHuman17Frames);
  viewer.setPose(allHuman17Frames[0]);
  viewer.resize();
</script>
```

方法约定：

- `setSequence(poses)`：从序列估计稳定体段尺寸，返回 profile；它不负责逐帧播放。
- `setPose(pose, render=true)`：设置单帧 17 点；也兼容传入 39 点并转交 `setExtendedPose`。形状错误时隐藏模型并返回 `false`。
- `setExtendedSequence(poses)`：用 39 点序列估计核心 17 点尺寸，并设置首帧。
- `setExtendedPose(pose, valid=null, render=true)`：设置单帧 39 点和有效掩码；无掩码时按有限数值推断。
- `setNodeOverlayData(data, render=true)`：传入 selected/full 节点坐标、有效掩码和分数，返回当前可见节点数。
- `getVisibleNodeCount()`：读取诊断叠加层的可见节点数。
- `setView(nextView, render=true)`：设置 `yaw`、`pitch`、`zoom`、`skeletonScale`。
- `setOptions(nextOptions, render=true)`：设置区域、骨架、扩展几何、节点模式、阈值和粗细。
- `resize()`：canvas 容器尺寸变化后调用。
- `render()`：手动重绘。
- `dispose()`：页面销毁或重建查看器前调用。

`setOptions` 支持：

```javascript
viewer.setOptions({
  regionsVisible: true,
  skeletonVisible: true,
  thickness: 1.0,
  extendedVisible: true,
  feetVisible: true,
  handsVisible: true,
  faceMarkersVisible: true,
  nodeMode: "selected",
  nodeThreshold: 0.25,
});
```

生产页面逐帧更新时，可以把 `render=false` 传给多个 setter，最后只调用一次 `render()`。

## 7. 17/39/133 点数据协议

### 核心 17 点

顺序固定为：

```text
0 root
1 right_hip       2 right_knee       3 right_ankle
4 left_hip        5 left_knee        6 left_ankle
7 spine           8 thorax           9 nose            10 head
11 left_shoulder  12 left_elbow      13 left_wrist
14 right_shoulder 15 right_elbow     16 right_wrist
```

### 扩展 22 点

追加在核心 17 点之后：

```text
17 left_big_toe       18 left_small_toe      19 left_heel
20 right_big_toe      21 right_small_toe     22 right_heel
23 left_thumb_tip     24 left_index_tip      25 left_middle_tip
26 left_ring_tip      27 left_pinky_tip
28 right_thumb_tip    29 right_index_tip     30 right_middle_tip
31 right_ring_tip     32 right_pinky_tip
33 left_eye           34 right_eye           35 left_ear
36 right_ear          37 nose_tip            38 mouth_center
```

核心 `0–16` 始终来自 MotionAGFormer，不被 RTMW3D 覆盖；RTMW3D 只提供扩展点和诊断节点。

### 坐标变换

NPZ 使用 `root_relative_image_units`。生成网页时，`build_interactive_3d_viewer.py` 把每个点从 `(x, y, z)` 转为显示坐标 `(x, z, -y)`。自行嵌入 Region Model 时，应保证传入数据已处于查看器期望的显示坐标系。

### 缺失值

- Python 融合结果用 `NaN` 和 `extended_valid=false` 表示无效扩展点。
- 写入 HTML 时，无效点压缩为 `null`。
- 查看器隐藏依赖无效点的几何，不会因单点缺失中断播放。
- 定向脚/手不可用时回退到 17 点端点球；头部和胸部尽量使用核心点估计方向。

## 8. 上下游相关路径

### [入口] `scripts/run_video_pose_pipeline.py`

推荐总入口。它按顺序调用 MMPose、主运动员选择、可选 2D 修正、MotionAGFormer、可选 3D 修正、可选 RTMW3D、融合和 HTML 生成。

常用命令：

```powershell
python .\scripts\run_video_pose_pipeline.py `
  --name test `
  --device cuda:0 `
  --fps 60 `
  --extended-pose `
  --extended-node-mode selected
```

重要参数：

- `--video` 优先于 `--name` 和 `--video-dir`。
- `--output-name` 只改变输出文件名前缀。
- `--output-dir` 改变输出根目录，默认 `outputs/`。
- `--pred-json` 会复用现有 MMPose JSON，并自动跳过 MMPose。
- `--skip-mmpose`、`--skip-2d`、`--skip-3d`、`--skip-viewer` 复用或跳过阶段。
- `--refine-2d`、`--refine-3d` 打开可选修正。
- `--extended-pose` 打开 133 点推理与 39 点融合。
- `--skip-rtmw3d` 复用已有 WholeBody NPZ。
- `--extended-node-mode` 只控制页面初始诊断节点模式。
- `--node-confidence-threshold` 只控制诊断节点初始阈值。
- 默认用 `conda run -n mmpose python` 启动子命令；可用 `--python-exe` 或 `--no-conda-run` 覆盖。

注意：脚本在执行任何阶段前会检查视频、MMPose、MotionAGFormer、AP3D 权重和配置路径。因此即使跳过部分阶段，对应外部仓库仍需存在。

### [入口] `scripts/infer_rtmw3d_wholebody.py`

对视频逐帧执行 RTMW3D WholeBody 推理，并利用已有 H36M 2D 数据尽量保持主运动员身份一致。支持：

- `--resume` 从已保存的 NPZ 继续。
- `--save-every` 周期性原子保存，默认每 50 帧。
- `--bbox-thr` 控制检测框阈值。
- `--max-frames` 用于小规模调试。

注意：当前实现虽然声明了 `--kpt-thr`，但主流程尚未实际使用该参数，不能依赖它过滤输出点。

原始 NPZ 主要键：

```text
keypoints_3d_cam   (T, 133, 3)
keypoints_2d       (T, 133, 2)
keypoint_scores    (T, 133)
bboxes             (T, 4)
selected           (T,)
processed          (T,)
```

### [入口] `scripts/fuse_extended_pose.py`

读取 17 点三维 NPZ 和原始 133 点 NPZ，按最短帧数截断并调用融合工具，输出 39 点 NPZ 和可选统计 JSON。它保留旧键 `pred3d_root_relative_image_units`，所以旧 17 点消费者仍可工作。

### [核心] `utils/extended_pose_schema.py`

Python 侧协议定义：

- 17、39、133 点数量和名称。
- 39 点父节点和骨骼。
- H36M 与 WholeBody 的 13 个共享锚点。
- 22 个扩展点从 WholeBody-133 中的来源索引。
- 眼、鼻、嘴的聚合与回退索引。
- 扩展点提取、形状校验和 schema 元数据。

此文件应被视为 Python 侧协议的单一事实来源。

### [核心] `utils/extended_pose_fusion.py`

主要处理顺序：

1. 从 WholeBody-133 提取 22 个目标点。
2. 使用共享锚点估计相似变换。
3. 检查锚点数量、残差和尺度，拒绝不可靠帧。
4. 平滑旋转和尺度。
5. 把手、脚、脸按局部父节点重新附着到 H36M 核心。
6. 调用 `utils/pose_constraints.py` 对短缺口插值并进行 One Euro 平滑。
7. 输出 39 点、完整对齐 133 点、有效性、置信度和诊断数据。

融合 NPZ 的新增键：

```text
core_pose_3d                  (T, 17, 3)
extended_pose_3d              (T, 39, 3)
extended_scores               (T, 39)
extended_valid                (T, 39)
alignment_residual            (T,)
alignment_accepted            (T,)
extended_source_status        (T, 39)
extended_joint_names          (39,)
wholebody_pose_3d_aligned     (T, 133, 3)
wholebody_scores              (T, 133)
wholebody_valid               (T, 133)
extended_schema               JSON 字符串标量
extended_stats                JSON 字符串标量
```

同时保留源 3D NPZ 中的其他键。融合过程内部还会计算全局/逐帧旋转与尺度，但当前 `build_extended_npz_payload()` 不把这些数组写入最终 NPZ。统计详情同时存在于 `extended_stats` 和可选的 `*_h36m17_extended39.json`。

### [核心] `utils/pose_constraints.py`

为融合模块提供 `interpolate_gaps()` 和 `filter_sequence_one_euro()`。它不是 Region Model 专用文件，但扩展点短缺口修复与时序平滑直接依赖它；修改函数签名或有效掩码语义时必须运行扩展姿态测试。

### [配置] `configs/extended_pose_default.json`

默认融合参数：

- `score_threshold=0.25`：WholeBody 点最低置信度。
- `shared_anchor_min=6`：单帧对齐最少共享锚点。
- `residual_threshold=0.65`：对齐残差阈值。
- `scale_window=31`：尺度平滑窗口。
- `max_gap=8`：可修复的最大连续缺失帧数。
- `repaired_score=0.35`：插值点分数。
- `length_clip=[0.25, 2.8]`：局部长度裁剪范围。
- `one_euro`：时序平滑参数。

调整参数时，优先复制一份配置并用 `--extended-config` 指定，避免直接改变所有数据的默认行为。

已知一致性问题：配置文件和 `default_extended_config()` 的 `residual_threshold` 都是 `0.65`，但融合函数在该键缺失时使用的代码回退值是 `0.22`。自定义配置不要删除此键；后续维护时应统一这两个默认值。

### [入口] `scripts/build_interactive_3d_viewer.py`

读取 `pred3d_root_relative_image_units`；若存在，也读取 39 点和对齐后的 133 点。它会：

- 转换到 Three.js 显示坐标。
- 从视频抽左侧 JPG，或复用 `--left-frame-dir`。
- 将姿态数据、Region Model 构建包和页面逻辑全部内嵌到 HTML。
- 根据数据是否存在自动启用或降级扩展节点模式。
- 按最短可用帧数截断姿态与左侧图像。

输出 HTML 不引用 CDN，但左侧 JPG 仍是相对外部资源；移动 HTML 时应同时移动对应帧目录并保持相对关系。

完整 133 点会显著放大内嵌数据；生成文件超过 100 MB 时脚本会警告。长视频优先使用 `selected` 或 `hidden`。

### [配置] `.env.example`、`project_config.py`

`.env.example` 给出本机路径模板；复制为 `.env` 后配置：

```dotenv
MMPOSE_ROOT=../mmpose-main
MOTIONAGFORMER_ROOT=../MotionAGFormer-master
PIPELINE_VIDEO_NAME=test
PIPELINE_VIDEO_DIR=input_videos/finefs_test
```

`project_config.py` 不依赖 `python-dotenv`，会自行读取简单的 `KEY=VALUE`。相对路径都以仓库根目录解析。

### [说明] `docs/finefs_video_to_interactive_3d_pose.md`

更完整的视频流水线分步说明，包括每个阶段的独立命令、中间产物和可选固定视角视频。排查总入口时，先按该文档逐阶段运行。

### [入口] `scripts/run_corrected_pose_pipeline.py`

人工修正 2D 后重新完成 3D lifting、可选约束修正和交互 HTML 生成。它同样消费 `build_interactive_3d_viewer.py`，但当前不负责 RTMW3D 39 点扩展分支。

### [入口] `scripts/run_manual_3d_correction_pipeline.py`

把人工编辑后的 3D 结果转换为 NPZ 并重建交互 HTML。它是“修正后重新查看”的入口，同样依赖已构建的 Region Model bundle。

### [测试] `tests/test_extended_pose.py`

覆盖：

- 39 点数量、父节点和索引。
- 从 WholeBody-133 提取扩展点及面部回退。
- Umeyama 相似变换恢复。
- 核心 17 点保持不变。
- 低置信度点隐藏。
- 短缺口插值。
- 旧 17 点 NPZ 键兼容。

### [测试] `tests/test_interactive_3d_viewer.py`

覆盖：

- 17 点 HTML 构建。
- 39 点 payload。
- 完整 133 点模式。
- HTML 无占位符残留、无外部脚本/CDN。

### [测试] `tests/test_pose_constraints.py`

覆盖融合间接依赖的插值与 One Euro 平滑等通用约束工具。改动 `utils/pose_constraints.py` 时应与扩展姿态测试一起运行。

## 9. 输出目录与文件

默认输出根目录是 **[本地数据] `outputs/`**，已被仓库 `.gitignore` 忽略。

```text
outputs/
├─ mmpose_pred_<name>/
│  └─ <video-name>.json
├─ mmpose_vis_<name>/
│  └─ <video-name>.mp4
├─ processed_2d/
│  ├─ <name>_h36m.npz
│  ├─ <name>_h36m.json
│  └─ <name>_h36m_vis.mp4
├─ processed_3d/
│  ├─ <name>_ap3d_motionagformer.npz
│  ├─ <name>_wholebody133_raw.npz
│  ├─ <name>_wholebody133_raw.json
│  ├─ <name>_h36m17_extended39.npz
│  └─ <name>_h36m17_extended39.json
└─ interactive_3d/
   ├─ <name>_frames_left/
   │  └─ 0000.jpg ...
   └─ <name>_interactive_3d.html
```

启用 2D/3D 修正时还会出现 `_refined` 文件和 `_refine_compare.json` 质量报告。

## 10. 环境准备

### JavaScript

需要 Node.js 和 npm。安装后：

```powershell
Set-Location .\region_model
npm install
npm run build
```

### Python

`requirements.txt` 包含仓库常规依赖，但完整姿态流水线还依赖外部 MMPose/MotionAGFormer 环境及其 PyTorch、MMEngine、MMPose 等依赖。推荐使用独立的 `mmpose` Conda 环境。

### [外部] 必备路径

```text
<MMPOSE_ROOT>/
├─ demo/inferencer_demo.py
└─ projects/rtmpose3d/        # 仅扩展分支需要

<MOTIONAGFORMER_ROOT>/
└─ configs/h36m/MotionAGFormer-small.yaml

<repo>/test/
├─ motionagformer-s-ap3d.pth.tr
└─ rtmpose-x...pth
```

模型权重、视频、NPZ 和输出媒体均被 `.gitignore` 忽略，交接时必须通过共享存储另行提供，不能只交 Git 仓库。

## 11. 验证命令与当前结果

协议与融合测试：

```powershell
python -m unittest tests.test_extended_pose tests.test_pose_constraints
```

当前工作区验证结果：扩展姿态 7 项、通用约束 14 项，共 21 项通过。

HTML 生成测试：

```powershell
python -m unittest tests.test_interactive_3d_viewer
```

当前工作区所在路径含中文目录。在当前 Windows 终端/Python 子进程组合下，路径被解析显示为乱码，3 项测试都在启动 HTML 生成子进程时失败；这不能证明 HTML 逻辑本身失败。请先把仓库复制或克隆到纯英文路径（例如 `D:\work\tree`）后复测。

前端构建：

```powershell
Set-Location .\region_model
npm install
npm run build
```

当前机器未找到 npm，尚未本机复核构建。安装 Node.js 后必须补做。

## 12. 常见问题

### 页面报 Region Model bundle 缺失

先在 `region_model/` 运行 `npm install` 和 `npm run build`，确认 `dist/human17-region-model.js` 存在。

### 修改 `src/` 后页面没有变化

HTML 生成器读取的是 `dist`，不是 `src`。重新构建 JS，再重新生成 HTML。

### `selected` 或 `full` 自动变回 `hidden`

- `selected` 需要 NPZ 中存在 `extended_pose_3d`。
- `full` 需要存在 `wholebody_pose_3d_aligned`。
- 数据缺失时 HTML 生成器会警告并降级。

### 扩展点不显示

依次检查 `extended_valid`、`extended_scores`、页面 `Node confidence`、`Extended extras` 和各部位开关。诊断节点开关与定向头/手/脚几何开关彼此独立。

### 扩展分支失败但流水线显示完成

这是设计行为：RTMW3D 和融合阶段使用可选执行包装，失败后继续生成 17 点查看器。必须检查日志中的 `OPTIONAL STAGE FAILED`，并确认最终打印的 `3D NPZ` 实际指向 17 点还是 39 点文件。

### 跳过阶段后仍提示外部路径不存在

总入口会在开始时统一检查外部仓库和权重。若只想单独生成页面，直接执行 `scripts/build_interactive_3d_viewer.py`，不要经过总入口。

## 13. 修改时的同步清单

### 修改点位协议

同时检查：

```text
utils/extended_pose_schema.py
region_model/src/profile.js
utils/extended_pose_fusion.py
scripts/build_interactive_3d_viewer.py
tests/test_extended_pose.py
tests/test_interactive_3d_viewer.py
region_model/README.md
region_model/HANDOFF.md
```

### 修改查看器 API 或样式

1. 修改 `region_model/src/`。
2. 必要时修改 `scripts/build_interactive_3d_viewer.py` 的页面模板和调用。
3. 运行 `npm run build` 更新 `dist`。
4. 运行 HTML 生成测试。
5. 用真实 17 点和 39 点数据各打开一次页面。

### 修改融合算法或默认参数

1. 修改 `utils/extended_pose_fusion.py` 或配置。
2. 保证核心 17 点逐值不变。
3. 运行 `tests.test_extended_pose`。
4. 检查统计 JSON 的接受帧数、平均残差、扩展点有效率和插值数。
5. 用 `selected`/`full` 叠加模式肉眼检查对齐质量。

## 14. 正式交接前必须另行提供

- MMPose 和 MotionAGFormer 的可用版本或 commit。
- AP3D、RTMPose、RTMW3D 权重及校验值/下载地址。
- 一段可公开的最小测试视频。
- 对应的 17 点、133 点、39 点样例 NPZ。
- 一份已生成且确认可打开的交互 HTML 与配套帧目录。
- 已知效果较差的视频名单及原因。
- 在纯英文路径、具备 Node/npm 的机器上完成本文第 11 节全部验证后的结果。
