# FineFS Video to Interactive 3D Pose

本文档记录如何把一个 FineFS 原视频处理成可交互查看的 3D 骨架页面。当前推荐只保留交互式 HTML 作为最终检查入口，普通对比视频、抽帧图片、固定视角 cube 视频都作为本地中间结果，不提交到 git。

## 需要准备

1. Conda 环境：`mmpose`
2. MMPose 仓库：`https://github.com/open-mmlab/mmpose`
3. MotionAGFormer 仓库：`https://github.com/TaatiTeam/MotionAGFormer`
4. AP3D 权重：`https://github.com/calvinyeungck/AthletePose3D`
5. 待处理视频，例如：`input_videos\finefs_test\`

## 推荐：一键运行完整视频

如果只是想跑一个完整视频，不需要逐个修改多个 py 文件。推荐使用总入口脚本：

先在项目根目录的 `.env` 中配置每台电脑自己的外部仓库目录。该文件已被 `.gitignore` 忽略，不会在两台电脑之间互相覆盖：

```dotenv
MMPOSE_ROOT=../mmpose-main
MOTIONAGFORMER_ROOT=../MotionAGFormer-master
PIPELINE_VIDEO_NAME=test
```

```powershell
# 在项目根目录中运行
conda activate mmpose

python .\scripts\run_video_pose_pipeline.py `
  --name test `
  --fps 30 `
  --device cpu
```

其中 `--name test` 对应输入视频：

```text
input_videos\finefs_test\test.mp4
```

如果要切换测试视频，优先使用 `--name`；也可以修改本机 `.env`：

```dotenv
PIPELINE_VIDEO_NAME=test
```

不建议修改任何 Python 文件里的本机路径，因为这些功能脚本应保持通用。

可选：在 17 点核心骨架上并行融合 RTMW3D 全身扩展点（脚、指尖、面部方向），生成 39 点查看结果。需要本机 MMPose 包含 `projects/rtmpose3d`：

```powershell
python .\scripts\run_video_pose_pipeline.py `
  --name test `
  --fps 60 `
  --device cuda:0 `
  --extended-pose
```

扩展点失败时仍会留下原来的 17 点 NPZ 和交互页面。原始 133 点与融合 39 点分别写入：

```text
outputs\processed_3d\<name>_wholebody133_raw.npz
outputs\processed_3d\<name>_h36m17_extended39.npz
```

## 运行一个新视频

在 PowerShell 中进入项目根目录：

```powershell
$ROOT = (Get-Location).Path
$NAME = "sample2"
$VIDEO = "$ROOT\input_videos\finefs_test\$NAME.mp4"
$MMP = "<MMPose 仓库目录>"
$MAG = "<MotionAGFormer 仓库目录>"
$FPS = 29
```

如果你已经激活环境：

```powershell
conda activate mmpose
```

下面命令里的 `python` 可以直接运行。如果不想激活环境，也可以把每条命令开头的 `python` 替换成 `conda run -n mmpose python`。

## 1. MMPose 提取 2D 骨架

```powershell
python "$MMP\demo\inferencer_demo.py" "$VIDEO" `
  --pose2d body `
  --device cpu `
  --show-progress `
  --pred-out-dir "$ROOT\outputs\mmpose_pred_$NAME" `
  --vis-out-dir "$ROOT\outputs\mmpose_vis_$NAME"
```

主要输出：

```text
outputs\mmpose_pred_sample2\sample2.json
outputs\mmpose_vis_sample2\sample2.mp4
```

其中 `json` 会进入下一步；`mp4` 只是 MMPose 自带可视化，可删除。

## 2. 选择主滑冰者并转 H36M 17 点

```powershell
python "$ROOT\scripts\select_main_skater_h36m.py" `
  --pred-json "$ROOT\outputs\mmpose_pred_$NAME\$NAME.json" `
  --video "$VIDEO" `
  --out-npz "$ROOT\outputs\processed_2d\${NAME}_h36m.npz" `
  --out-json "$ROOT\outputs\processed_2d\${NAME}_h36m.json" `
  --vis-out "$ROOT\outputs\processed_2d\${NAME}_h36m_vis.mp4"
```

主要输出：

```text
outputs\processed_2d\sample2_h36m.npz
outputs\processed_2d\sample2_h36m.json
outputs\processed_2d\sample2_h36m_vis.mp4
```

其中 `npz` 会进入 3D lifting；`sample2_h36m_vis.mp4` 会用于交互页面左侧视频帧。

## 3. AP3D / MotionAGFormer 从 2D 升到 3D

```powershell
python "$ROOT\scripts\lift_2d_to_3d_motionagformer_ap3d.py" `
  --input-2d-npz "$ROOT\outputs\processed_2d\${NAME}_h36m.npz" `
  --checkpoint "$ROOT\test\motionagformer-s-ap3d.pth.tr" `
  --motionagformer-root "$MAG" `
  --config "$MAG\configs\h36m\MotionAGFormer-small.yaml" `
  --out-npz "$ROOT\outputs\processed_3d\${NAME}_ap3d_motionagformer.npz" `
  --out-json "$ROOT\outputs\processed_3d\${NAME}_ap3d_motionagformer.json" `
  --vis-out "$ROOT\outputs\processed_3d\${NAME}_ap3d_motionagformer_vis.mp4" `
  --device cpu
```

主要输出：

```text
outputs\processed_3d\sample2_ap3d_motionagformer.npz
outputs\processed_3d\sample2_ap3d_motionagformer.json
outputs\processed_3d\sample2_ap3d_motionagformer_vis.mp4
```

交互页面只需要 `npz`。`json` 方便人工检查；`vis.mp4` 是固定视角可视化，可删除。

## 4. 生成交互式 3D 页面

推荐直接从 2D H36M 可视化视频抽左侧帧，不再单独生成普通 frame compare 页面：

```powershell
python "$ROOT\scripts\build_interactive_3d_viewer.py" `
  --input-3d-npz "$ROOT\outputs\processed_3d\${NAME}_ap3d_motionagformer.npz" `
  --left-video "$ROOT\outputs\processed_2d\${NAME}_h36m_vis.mp4" `
  --extract-frame-dir "$ROOT\outputs\interactive_3d\${NAME}_frames_left" `
  --out-html "$ROOT\outputs\interactive_3d\${NAME}_interactive_3d.html" `
  --title "$NAME Interactive 2D / 3D Skeleton" `
  --fps $FPS
```

最终主要查看：

```text
outputs\interactive_3d\sample2_interactive_3d.html
```

这个页面左侧是 2D 标注帧，右侧是可旋转、缩放、逐帧播放的 3D 骨架坐标空间。播放时两侧共用同一个帧号。

## 可选：固定视角方格立方体视频

如果需要导出一个固定视角的 3D cube 视频，可以运行：

```powershell
python "$ROOT\scripts\render_3d_skeleton_cube.py" `
  --input-3d-npz "$ROOT\outputs\processed_3d\${NAME}_ap3d_motionagformer.npz" `
  --out-video "$ROOT\outputs\processed_3d\${NAME}_ap3d_motionagformer_cube_grid.mp4" `
  --out-frame-dir "$ROOT\outputs\processed_3d\cube_grid_frames_$NAME" `
  --fps $FPS `
  --grid-steps 8 `
  --keep-frames
```

这是可选输出；如果只保留交互页面，可以不运行。

## 建议提交到 git 的文件

建议提交这些脚本：

```text
scripts/select_main_skater_h36m.py
scripts/lift_2d_to_3d_motionagformer_ap3d.py
scripts/build_interactive_3d_viewer.py
```

可选提交：

```text
scripts/render_3d_skeleton_cube.py
scripts/make_video_clip.py
```

不建议提交，除非后面还想保留普通同步页面：

```text
scripts/build_frame_compare.py
scripts/build_frame_dir_compare.py
```

## 可以删除的本地生成文件

如果只需要交互式页面，以下目录和文件都可以删除，之后需要时重新生成：

```text
outputs\mmpose_vis_*
outputs\comparison_frames\
outputs\comparison\
outputs\comparison_cube_frames\
outputs\processed_3d\*_vis.mp4
outputs\processed_3d\*_cube*.mp4
outputs\processed_3d\cube*_frames_*
```

这些也可以删除，但删除后重新跑流程会耗时：

```text
outputs\mmpose_pred_*
outputs\processed_2d\
outputs\processed_3d\*.npz
outputs\processed_3d\*.json
```

输入视频和模型权重通常不提交到 git，但本地建议保留：

```text
input_videos\
test\motionagformer-s-ap3d.pth.tr
```

如果不再使用旧的 AthletePose3D 2D 模型，可以删除：

```text
test\moganet_b_ap2d_384x288.pth
```

论文 PDF 和临时相机参数文件也不需要提交：

```text
test\*.pdf
test\cam_param.json
```

## `.gitignore` 应忽略

当前建议忽略：

```text
input_videos/
outputs/
test/
*.mp4
*.avi
*.mov
*.mkv
*.webm
*.pth
*.pth.tr
*.pt
*.ckpt
*.onnx
```

这样 git 只保留代码和说明文档，不会把视频、权重、抽帧图片、推理结果提交进去。
