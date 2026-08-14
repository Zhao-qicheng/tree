# 2D H36M 骨架手动修正流程

这个工具用于修正已经生成的 H36M-17 二维骨架点，然后重新生成 3D 骨架。

## 1. 生成修正页面

示例：

```powershell
# 在项目根目录中运行
conda activate mmpose

python .\scripts\build_2d_pose_editor.py `
  --video ".\input_videos\finefs_test\test2.mp4" `
  --h36m-json ".\outputs\processed_2d\test2_h36m.json" `
  --source-npz ".\outputs\processed_2d\test2_h36m.npz" `
  --out-html ".\outputs\pose_editor\test2_2d_pose_editor.html"
```

打开生成的 HTML 后，可以拖拽关节点。保存时建议保存到：

```text
outputs\corrected_json\test2_corrected_h36m.json
```

## 2. 重新生成 3D

```powershell
python .\scripts\run_corrected_pose_pipeline.py `
  --name test2 `
  --corrected-json ".\outputs\manual_edits\test2_corrected_h36m.json" `
  --device cpu `
  --fps 30
```

输出位置：

```text
outputs\manual_corrected_2d\
outputs\manual_corrected_3d\
outputs\interactive_3d\
```

## 建议

不需要每一帧都修。优先修关键帧，例如起跳、腾空、落冰、遮挡严重或左右肢体混乱的帧。修完关键帧后，在页面里使用 `Interp Joint` 或 `Interp All` 生成中间帧。

这个流程不会覆盖原来的自动标注结果，便于比较自动版和人工修正版。
