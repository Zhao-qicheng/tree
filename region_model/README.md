# Human17 Region Model

`region_model` turns Human3.6M / Human17 `[17][3]` poses into a lightweight
rigid-body visual model: capsule limbs, a tapered torso, a head ellipsoid, and
terminal hand/foot spheres. It is a viewer-side visualisation module, not a
physics, collision, MuJoCo, or PyBullet model.

## Build

```powershell
cd region_model
npm install
npm run build
```

The resulting `dist/human17-region-model.js` is an IIFE bundle that includes
Three.js. It has no CDN dependency and can be loaded from a local HTML file.

## Browser API

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

`setSequence()` samples the pose sequence and derives one stable set of body
radii using robust medians. This prevents noisy 3D detections from changing the
body thickness on every frame. `setPose()` accepts only Human17 point arrays;
individual invalid points hide their affected body parts instead of stopping
playback.

Human17 has no toe, heel, palm, or finger landmarks. The first profile uses
spheres at ankle and wrist endpoints and deliberately does not infer those
unobserved directions.
