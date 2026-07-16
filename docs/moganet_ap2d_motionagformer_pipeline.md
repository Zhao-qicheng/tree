# MogaNet AP2D to MotionAGFormer Pipeline

这条流程用于和当前 RTMPose 流程做对比。它保留同一个人体检测框来源，然后把 2D 关键点模型换成 AthletePose3D AP2D 的 MogaNet 权重，最后仍然使用 MotionAGFormer 生成 3D 骨架。

## 流程

```text
FineFS video
 -> RTMDet bbox source from MMPose inferencer
 -> MogaNet AP2D 2D keypoints
 -> select_main_skater_h36m.py
 -> MotionAGFormer AP3D 2D-to-3D
 -> interactive 3D HTML
```

这里第 1 步仍会调用 MMPose inferencer，但后续只使用其中的 `bbox`。真正用于 3D 的 2D 关键点来自 `moganet_b_ap2d_384x288.pth`。

## 需要准备

```text
C:\Users\86158\Desktop\MogaNet-main
C:\Users\86158\Desktop\MotionAGFormer-master
C:\Users\86158\Desktop\mmpose-main
八叉树\test\moganet_b_ap2d_384x288.pth
八叉树\test\motionagformer-s-ap3d.pth.tr
八叉树\input_videos\finefs_test\<name>.mp4
```

当前实现是 standalone MogaNet 推理，不需要安装旧版 MMPose 0.29。它会复用 `MogaNet-main\models\moganet.py` 中的 backbone 代码，并手动完成 crop、resize、heatmap decode。

## 一键运行

```powershell
cd C:\Users\86158\Desktop\八叉树
conda activate mmpose

python .\scripts\run_moganet_pose_pipeline.py `
  --name test1 `
  --video-dir .\input_videos\finefs_test `
  --mmpose-root C:\Users\86158\Desktop\mmpose-main `
  --moganet-root C:\Users\86158\Desktop\MogaNet-main `
  --motionagformer-root C:\Users\86158\Desktop\MotionAGFormer-master `
  --device cpu `
  --fps 25
```

如果只想先试 100 帧：

```powershell
python .\scripts\run_moganet_pose_pipeline.py `
  --name test1 `
  --limit-frames 100 `
  --device cpu
```

## 主要输出

```text
outputs\moganet_pred_<name>\<name>.json
outputs\moganet_vis_<name>\<name>.mp4
outputs\processed_2d\<name>_moganet_h36m.npz
outputs\processed_2d\<name>_moganet_h36m_vis.mp4
outputs\processed_3d\<name>_moganet_ap3d_motionagformer.npz
outputs\interactive_3d\<name>_moganet_interactive_3d.html
```

对比当前 RTMPose 分支时，主要看这两个页面：

```text
outputs\interactive_3d\<name>_interactive_3d.html
outputs\interactive_3d\<name>_moganet_interactive_3d.html
```

## 复用已有检测框

如果已经跑过当前 RTMPose 主流程，通常已有：

```text
outputs\mmpose_pred_<name>\<name>.json
```

这时可以跳过第 1 步：

```powershell
python .\scripts\run_moganet_pose_pipeline.py `
  --name test1 `
  --skip-bbox `
  --device cpu
```

## 单独运行 MogaNet 2D

```powershell
python .\scripts\infer_moganet_ap2d_from_bboxes.py `
  --video .\input_videos\finefs_test\test1.mp4 `
  --bbox-json .\outputs\mmpose_pred_test1\test1.json `
  --out-json .\outputs\moganet_pred_test1\test1.json `
  --vis-out .\outputs\moganet_vis_test1\test1.mp4 `
  --moganet-root C:\Users\86158\Desktop\MogaNet-main `
  --checkpoint .\test\moganet_b_ap2d_384x288.pth `
  --device cpu
```

## 注意

MogaNet-B 在 CPU 上会比较慢，完整视频可能需要较长时间。建议先用 `--limit-frames 100` 检查效果，再跑完整视频。

MogaNet 分支和 RTMPose 分支使用同一个人体框来源，这样比较时主要差异来自 2D 姿态模型本身，更适合判断哪一个更贴合主滑冰者。
