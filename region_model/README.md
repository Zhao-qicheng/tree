# Human17 Region Model

`region_model` 把 Human3.6M / Human17 的 `[17][3]` 姿态渲染成轻量刚体可视化模型：胶囊四肢、锥台躯干、头部椭球，以及手腕/脚踝端点球。它只用于查看，不是物理、碰撞、MuJoCo 或 PyBullet 模型。

第一版还支持可选的 **H36M-17 + 扩展 22 点**（共 39 点）。扩展点来自 RTMW3D COCO-WholeBody 133 点筛选：双脚 6 点、双手指尖 10 点、面部方向 6 点。核心 17 点始终保持 MotionAGFormer 结果，Region Model 用扩展点生成朝向可辨的头、手、脚。

## 构建

```powershell
cd region_model
npm install
npm run build
```

产物 `dist/human17-region-model.js` 是包含 Three.js 的 IIFE 包，无 CDN 依赖，可直接嵌入本地 HTML。

## 浏览器 API

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
  viewer.setOptions({ regionsVisible: true, skeletonVisible: true, thickness: 1.0 });
  viewer.setView({ yaw: 0, pitch: 0, zoom: 1.15, skeletonScale: 1 });
  viewer.resize();
</script>
```

`setSequence()` 从序列中估计一套稳定体段半径。`setPose()` 仍只接受 Human17 的 17 点数组；无效点会隐藏对应部件而不是中断播放。

如果有融合后的 39 点：

```javascript
viewer.setExtendedSequence(allExtendedFrames);
viewer.setExtendedPose(allExtendedFrames[0], validMask);
viewer.setOptions({
  extendedVisible: true,
  feetVisible: true,
  handsVisible: true,
  faceMarkersVisible: true,
});
```

扩展点缺失时，定向脚/手会隐藏并回退到原来的端点球；胸部朝向仍可从肩部和躯干估计。

## 39 点协议

顺序为：H36M 核心 17 点，随后

- 17–22：左/右 大脚趾、小脚趾、脚跟
- 23–32：左/右 拇指、食指、中指、无名指、小指指尖
- 33–38：左眼、右眼、左耳、右耳、鼻尖、嘴部中心

融合结果写入独立 NPZ：`core_pose_3d`、`extended_pose_3d`、`extended_scores`、`extended_valid`，同时保留原来的 `pred3d_root_relative_image_units`，旧查看器和检索流程可继续只读 17 点。

## 流水线

需要本机 MMPose 带 `projects/rtmpose3d`，然后在总入口打开扩展分支：

```powershell
python scripts/run_video_pose_pipeline.py `
  --name test `
  --device cuda:0 `
  --fps 60 `
  --extended-pose `
  --extended-node-mode selected
```

原始 133 点保存为 `*_wholebody133_raw.npz`，融合 39 点保存为 `*_h36m17_extended39.npz`。RTMW3D 失败时流水线仍会生成原来的 17 点结果和查看器。

`--extended-node-mode` 控制页面打开时的额外节点显示状态：

- `hidden`：默认隐藏，只显示 Region Model 和原有17点骨架。
- `selected`：显示融合后的22个脚、指尖和面部方向点。
- `full`：显示对齐后的完整RTMW3D 133点。

页面生成后仍可通过 `RTMW3D nodes` 下拉框切换三种模式，并使用
`Node confidence` 调整显示阈值。该节点叠加层只用于检查推理质量，
与控制定向头、手、脚几何的 `Extended extras` 开关相互独立。
