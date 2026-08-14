# 3D 骨架手动修正流程

这个工具用于修正已经由 MotionAGFormer 生成的 H36M-17 三维骨架。流程是：

`原 3D NPZ -> 打开 3D 修正页面 -> 保存 corrected JSON -> 转回 corrected NPZ -> 重新生成交互 3D 页面`

## 1. 生成 3D 修正页面

默认视频名由本机 `.env` 中的 `PIPELINE_VIDEO_NAME` 设置。假设该值为 `test1`，默认读取：

- `outputs/processed_3d/test1_ap3d_motionagformer.npz`
- `outputs/processed_2d/test1_h36m_vis.mp4`
- 输出到 `outputs/pose3d_editor/test1_3d_pose_editor.html`

运行：

```powershell
# 在项目根目录中运行
conda activate mmpose

python .\scripts\build_3d_pose_editor.py
```

如果要长期切换本机默认视频，修改不提交到 Git 的 `.env`：

```dotenv
PIPELINE_VIDEO_NAME=test2
```

也可以临时用命令行参数指定输入和输出，不需要修改 Python 文件：

```powershell
python .\scripts\build_3d_pose_editor.py `
  --input-3d-npz ".\outputs\processed_3d\test2_ap3d_motionagformer.npz" `
  --left-video ".\outputs\processed_2d\test2_h36m_vis.mp4" `
  --out-html ".\outputs\pose3d_editor\test2_3d_pose_editor.html" `
  --title "test2 3D Pose Editor"
```

## 2. 页面里怎么修

左侧是参考视频帧，右侧是 3D 骨架修正区。三维坐标显示为：

- `X`：左右方向
- `Depth`：前后深度
- `Height`：上下高度

推荐修正顺序：

1. 用底部时间轴定位到错误帧。
2. 在右侧关节列表选择需要修正的关节。
3. 在 `Front / Side / Top` 三个正交视图中拖动关节点。
4. 如果拖动不够精细，用 `X / Depth / Height` 数值框或 `Nudge` 微调。
5. 只修关键帧，再用 `Interp Joint` 对当前关节做关键帧插值。
6. 如果整帧都偏了，可以用 `Copy Prev Frame` 或 `Copy Next Frame` 复制邻近帧。
7. 修完后点击 `Save JSON` 或 `Download` 保存修正结果。

常用按钮：

- `Save JSON`：保存修正后的 3D JSON。
- `Download`：浏览器不支持文件选择器时，直接下载 JSON。
- `Undo`：撤销上一次修改。
- `Reset Frame`：把当前帧恢复成原始 3D 结果。
- `Copy Prev Joint / Copy Next Joint`：把相邻帧同一个关节复制到当前帧。
- `Copy Prev Frame / Copy Next Frame`：把相邻整帧骨架复制到当前帧。
- `Interp Joint`：用当前关节已经修过的关键帧插值中间帧。
- `Interp All`：对所有关节执行插值。
- `Prev Edit / Next Edit`：跳到上一个或下一个已经修改过的帧。
- `Skeleton + / Skeleton -`：只改变显示大小，不改变导出的 3D 坐标。
- `Zoom + / Zoom -`：缩放当前视图，也不改变导出坐标。

## 3. 把修正 JSON 转回 3D 数据

假设你在页面中保存了：

`outputs/manual_edits/test1_corrected_3d.json`

运行：

```powershell
python .\scripts\run_manual_3d_correction_pipeline.py `
  --name test1 `
  --corrected-json ".\outputs\manual_edits\test1_corrected_3d.json" `
  --source-3d-npz ".\outputs\processed_3d\test1_ap3d_motionagformer.npz" `
  --left-video ".\outputs\processed_2d\test1_h36m_vis.mp4" `
  --fps 30
```

输出位置：

- corrected NPZ：`outputs/manual_corrected_3d/test1_manual3d_corrected_3d.npz`
- corrected JSON：`outputs/manual_corrected_3d/test1_manual3d_corrected_3d.json`
- corrected 3D 视频：`outputs/manual_corrected_3d/test1_manual3d_corrected_3d_vis.mp4`
- 交互页面：`outputs/interactive_3d/test1_manual3d_interactive_3d.html`

## 4. 注意

3D 编辑器里显示的是 `display_x_depth_height` 坐标，转换脚本会自动换回 MotionAGFormer 输出 NPZ 使用的源坐标。因此你在页面里按直觉修 `X / Depth / Height` 即可。

这个工具适合修明显错误的 3D 姿态，例如腿部前后深度反了、手臂高度明显错了、跳跃姿态在几帧里闪动。它不适合逐帧完全重标整个视频，长视频更推荐“修关键帧 + 插值 + 重新生成 3D 页面检查”。
