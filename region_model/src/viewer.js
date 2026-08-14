import * as THREE from "three";

import {
  DEFAULT_PALETTE,
  HUMAN17_BONES,
  estimateHuman17Profile,
  isFinitePoint,
  isHuman17Pose,
} from "./profile.js";

const EPSILON = 1e-5;
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const FORWARD_AXIS = new THREE.Vector3(0, 0, 1);

function colorForBone(pair) {
  const key = pair.join("-");
  if (["0-4", "4-5", "5-6", "8-11", "11-12", "12-13"].includes(key)) return DEFAULT_PALETTE.left;
  if (["0-1", "1-2", "2-3", "8-14", "14-15", "15-16"].includes(key)) return DEFAULT_PALETTE.right;
  return DEFAULT_PALETTE.center;
}

export function alignGroupBetween(group, start, end) {
  const direction = end.clone().sub(start);
  const length = direction.length();
  if (!Number.isFinite(length) || length <= EPSILON) return null;
  group.position.copy(start).add(end).multiplyScalar(0.5);
  group.quaternion.setFromUnitVectors(Y_AXIS, direction.multiplyScalar(1 / length));
  return length;
}

class CapsulePart {
  constructor(parent, material, segments = 14) {
    this.group = new THREE.Group();
    this.cylinder = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, segments), material);
    this.startCap = new THREE.Mesh(new THREE.SphereGeometry(1, segments, Math.max(8, Math.floor(segments * 0.7))), material);
    this.endCap = new THREE.Mesh(this.startCap.geometry, material);
    this.group.add(this.cylinder, this.startCap, this.endCap);
    parent.add(this.group);
  }

  update(start, end, radius) {
    const length = alignGroupBetween(this.group, start, end);
    if (length === null || !Number.isFinite(radius) || radius <= EPSILON) {
      this.group.visible = false;
      return;
    }

    const effectiveRadius = Math.min(radius, length / 2);
    const cylinderLength = Math.max(0, length - effectiveRadius * 2);
    this.cylinder.visible = cylinderLength > EPSILON;
    this.cylinder.scale.set(effectiveRadius, cylinderLength, effectiveRadius);
    this.startCap.position.set(0, -cylinderLength / 2, 0);
    this.endCap.position.set(0, cylinderLength / 2, 0);
    this.startCap.scale.setScalar(effectiveRadius);
    this.endCap.scale.setScalar(effectiveRadius);
    this.group.visible = true;
  }
}

class TaperedCylinderPart {
  constructor(parent, material) {
    this.group = new THREE.Group();
    this.mesh = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 16), material);
    this.group.add(this.mesh);
    parent.add(this.group);
    this.baseLength = 1;
  }

  setShape(topRadius, bottomRadius, baseLength) {
    this.baseLength = Math.max(baseLength, EPSILON);
    this.mesh.geometry.dispose();
    this.mesh.geometry = new THREE.CylinderGeometry(
      Math.max(topRadius / this.baseLength, EPSILON),
      Math.max(bottomRadius / this.baseLength, EPSILON),
      1,
      18,
    );
  }

  update(start, end, thickness) {
    const length = alignGroupBetween(this.group, start, end);
    if (length === null) {
      this.group.visible = false;
      return;
    }
    const width = this.baseLength * thickness;
    this.mesh.scale.set(width, length, width);
    this.group.visible = true;
  }
}

class EllipsoidPart {
  constructor(parent, material) {
    this.group = new THREE.Group();
    this.mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 14), material);
    this.group.add(this.mesh);
    parent.add(this.group);
  }

  update(center, direction, radius) {
    if (!center || !direction || direction.lengthSq() <= EPSILON || !Number.isFinite(radius) || radius <= EPSILON) {
      this.group.visible = false;
      return;
    }
    this.group.position.copy(center);
    this.group.quaternion.setFromUnitVectors(Y_AXIS, direction.clone().normalize());
    this.mesh.scale.set(radius * 0.9, radius * 1.12, radius);
    this.group.visible = true;
  }
}

function createAxisLine(parent, endpoint, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), endpoint]);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 });
  const line = new THREE.Line(geometry, material);
  parent.add(line);
}

function addReferenceCube(parent, radius) {
  const size = radius * 2;
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x5f6871, transparent: true, opacity: 0.82 });
  const edgeGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(size, size, size));
  parent.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

  const gridColor = 0xb6bec6;
  const subduedGridColor = 0xd8dde3;
  const floor = new THREE.GridHelper(size, 8, gridColor, subduedGridColor);
  floor.rotation.x = Math.PI / 2;
  floor.position.z = -radius;
  floor.material.transparent = true;
  floor.material.opacity = 0.36;
  parent.add(floor);

  const back = new THREE.GridHelper(size, 8, gridColor, subduedGridColor);
  back.position.y = radius;
  back.material.transparent = true;
  back.material.opacity = 0.32;
  parent.add(back);

  const side = new THREE.GridHelper(size, 8, gridColor, subduedGridColor);
  side.rotation.z = Math.PI / 2;
  side.position.x = -radius;
  side.material.transparent = true;
  side.material.opacity = 0.25;
  parent.add(side);

  const axisLength = radius * 0.72;
  createAxisLine(parent, new THREE.Vector3(axisLength, 0, 0), 0xef4444);
  createAxisLine(parent, new THREE.Vector3(0, axisLength, 0), 0x2563eb);
  createAxisLine(parent, new THREE.Vector3(0, 0, axisLength), 0x16a34a);
}

export class Human17RegionViewer {
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("createHuman17RegionViewer requires a canvas element.");
    }

    this.canvas = canvas;
    this.radius = Math.max(Number(options.radius) || 1, EPSILON);
    this.pairs = options.pairs || HUMAN17_BONES;
    this.boneColors = options.boneColors || this.pairs.map(colorForBone);
    this.view = {
      yaw: Number(options.yaw) || 0,
      pitch: Number(options.pitch) || 0,
      zoom: Number(options.zoom) || 1,
      skeletonScale: Number(options.skeletonScale) || 1,
    };
    this.options = {
      regionsVisible: options.regionsVisible !== false,
      skeletonVisible: options.skeletonVisible !== false,
      thickness: Number(options.thickness) || 1,
    };
    this.profile = estimateHuman17Profile([]);
    this.currentPose = null;
    this.points = Array.from({ length: 17 }, () => new THREE.Vector3());
    this.valid = Array.from({ length: 17 }, () => false);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(options.background || "#f7f8fa", 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, this.radius * 12);
    this.camera.position.set(0, -this.radius * 4, 0);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(0, 0, 0);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x93a4b4, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
    keyLight.position.set(this.radius * 1.5, -this.radius * 2.2, this.radius * 2.7);
    this.scene.add(keyLight);

    this.pitchGroup = new THREE.Group();
    this.yawGroup = new THREE.Group();
    this.referenceGroup = new THREE.Group();
    this.poseGroup = new THREE.Group();
    this.regionGroup = new THREE.Group();
    this.skeletonGroup = new THREE.Group();
    this.pitchGroup.add(this.yawGroup);
    this.yawGroup.add(this.referenceGroup, this.poseGroup);
    this.poseGroup.add(this.regionGroup, this.skeletonGroup);
    this.scene.add(this.pitchGroup);
    addReferenceCube(this.referenceGroup, this.radius);

    this.materials = {
      center: new THREE.MeshStandardMaterial({ color: DEFAULT_PALETTE.center, roughness: 0.56, metalness: 0.04 }),
      left: new THREE.MeshStandardMaterial({ color: DEFAULT_PALETTE.left, roughness: 0.56, metalness: 0.04 }),
      right: new THREE.MeshStandardMaterial({ color: DEFAULT_PALETTE.right, roughness: 0.56, metalness: 0.04 }),
      head: new THREE.MeshStandardMaterial({ color: DEFAULT_PALETTE.head, roughness: 0.5, metalness: 0.05 }),
    };
    this._buildRegionParts();
    this._buildSkeletonOverlay();
    this._applyView();
    this._applyOptions();
  }

  _buildRegionParts() {
    const center = this.materials.center;
    this.capsules = {
      pelvis: new CapsulePart(this.regionGroup, center),
      shoulders: new CapsulePart(this.regionGroup, center),
      neck: new CapsulePart(this.regionGroup, center),
      rightThigh: new CapsulePart(this.regionGroup, this.materials.right),
      rightShin: new CapsulePart(this.regionGroup, this.materials.right),
      leftThigh: new CapsulePart(this.regionGroup, this.materials.left),
      leftShin: new CapsulePart(this.regionGroup, this.materials.left),
      rightUpperArm: new CapsulePart(this.regionGroup, this.materials.right),
      rightLowerArm: new CapsulePart(this.regionGroup, this.materials.right),
      leftUpperArm: new CapsulePart(this.regionGroup, this.materials.left),
      leftLowerArm: new CapsulePart(this.regionGroup, this.materials.left),
    };
    this.torso = new TaperedCylinderPart(this.regionGroup, center);
    this.head = new EllipsoidPart(this.regionGroup, this.materials.head);
    const endpointGeometry = new THREE.SphereGeometry(1, 16, 12);
    this.endpoints = {
      rightHand: new THREE.Mesh(endpointGeometry, this.materials.right),
      leftHand: new THREE.Mesh(endpointGeometry, this.materials.left),
      rightFoot: new THREE.Mesh(endpointGeometry, this.materials.right),
      leftFoot: new THREE.Mesh(endpointGeometry, this.materials.left),
    };
    this.regionGroup.add(...Object.values(this.endpoints));
  }

  _buildSkeletonOverlay() {
    this.boneLines = this.pairs.map((pair, index) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(6, 3));
      const material = new THREE.LineBasicMaterial({
        color: this.boneColors[index] || colorForBone(pair),
        transparent: true,
        opacity: 0.95,
        depthTest: false,
        depthWrite: false,
      });
      const line = new THREE.Line(geometry, material);
      line.renderOrder = 10;
      this.skeletonGroup.add(line);
      return { line, pair };
    });

    const jointGeometry = new THREE.SphereGeometry(1, 12, 10);
    this.jointMarkers = Array.from({ length: 17 }, (_, index) => {
      const material = new THREE.MeshBasicMaterial({
        color: index === 0 ? DEFAULT_PALETTE.root : DEFAULT_PALETTE.skeleton,
        depthTest: false,
        depthWrite: false,
      });
      const marker = new THREE.Mesh(jointGeometry, material);
      marker.renderOrder = 11;
      this.skeletonGroup.add(marker);
      return marker;
    });
  }

  _configureProfile() {
    this.torso.setShape(
      this.profile.radii.torsoTop,
      this.profile.radii.torsoBottom,
      this.profile.lengths.torsoLength,
    );
  }

  _applyOptions() {
    this.regionGroup.visible = this.options.regionsVisible;
    this.skeletonGroup.visible = this.options.skeletonVisible;
  }

  _applyView() {
    this.yawGroup.rotation.z = -this.view.yaw;
    this.pitchGroup.rotation.x = this.view.pitch;
    this.poseGroup.scale.setScalar(this.view.skeletonScale);
  }

  _updateCamera() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width || this.canvas.clientWidth || 1);
    const height = Math.max(1, rect.height || this.canvas.clientHeight || 1);
    const viewHeight = (this.radius * 2.63) / Math.max(this.view.zoom, EPSILON);
    const viewWidth = viewHeight * (width / height);
    this.camera.left = -viewWidth / 2;
    this.camera.right = viewWidth / 2;
    this.camera.top = viewHeight / 2;
    this.camera.bottom = -viewHeight / 2;
    this.camera.updateProjectionMatrix();
  }

  _setPoint(index, pose) {
    const source = pose[index];
    const valid = isFinitePoint(source);
    this.valid[index] = valid;
    if (valid) this.points[index].set(Number(source[0]), Number(source[1]), Number(source[2]));
  }

  _updateCapsule(name, first, second, radius) {
    const capsule = this.capsules[name];
    if (!this.valid[first] || !this.valid[second]) {
      capsule.group.visible = false;
      return;
    }
    capsule.update(this.points[first], this.points[second], radius * this.options.thickness);
  }

  _updateEndpoint(name, joint, radius) {
    const endpoint = this.endpoints[name];
    if (!this.valid[joint]) {
      endpoint.visible = false;
      return;
    }
    endpoint.position.copy(this.points[joint]);
    endpoint.scale.setScalar(radius * this.options.thickness);
    endpoint.visible = true;
  }

  _updateRegionGeometry() {
    const radii = this.profile.radii;
    this._updateCapsule("pelvis", 1, 4, radii.pelvis);
    this._updateCapsule("shoulders", 11, 14, radii.shoulders);
    this._updateCapsule("neck", 8, 9, radii.neck);
    this._updateCapsule("rightThigh", 1, 2, radii.rightThigh);
    this._updateCapsule("rightShin", 2, 3, radii.rightShin);
    this._updateCapsule("leftThigh", 4, 5, radii.leftThigh);
    this._updateCapsule("leftShin", 5, 6, radii.leftShin);
    this._updateCapsule("rightUpperArm", 14, 15, radii.rightUpperArm);
    this._updateCapsule("rightLowerArm", 15, 16, radii.rightLowerArm);
    this._updateCapsule("leftUpperArm", 11, 12, radii.leftUpperArm);
    this._updateCapsule("leftLowerArm", 12, 13, radii.leftLowerArm);

    if (this.valid[0] && this.valid[8]) this.torso.update(this.points[0], this.points[8], this.options.thickness);
    else this.torso.group.visible = false;

    const headCenter = this.valid[9] && this.valid[10]
      ? this.points[9].clone().lerp(this.points[10], 0.55)
      : this.valid[10] ? this.points[10].clone() : null;
    const headDirection = this.valid[9] && this.valid[10]
      ? this.points[10].clone().sub(this.points[9])
      : this.valid[8] && this.valid[9] ? this.points[9].clone().sub(this.points[8]) : FORWARD_AXIS;
    this.head.update(headCenter, headDirection, radii.head * this.options.thickness);

    this._updateEndpoint("rightHand", 16, radii.hand);
    this._updateEndpoint("leftHand", 13, radii.hand);
    this._updateEndpoint("rightFoot", 3, radii.foot);
    this._updateEndpoint("leftFoot", 6, radii.foot);
  }

  _updateSkeletonOverlay() {
    this.boneLines.forEach(({ line, pair }) => {
      const [first, second] = pair;
      const valid = this.valid[first] && this.valid[second];
      line.visible = valid;
      if (!valid) return;
      const positions = line.geometry.attributes.position;
      positions.setXYZ(0, this.points[first].x, this.points[first].y, this.points[first].z);
      positions.setXYZ(1, this.points[second].x, this.points[second].y, this.points[second].z);
      positions.needsUpdate = true;
    });

    const markerRadius = this.profile.radii.joint * this.options.thickness;
    this.jointMarkers.forEach((marker, index) => {
      marker.visible = this.valid[index];
      if (!marker.visible) return;
      marker.position.copy(this.points[index]);
      marker.scale.setScalar(index === 0 ? markerRadius * 1.35 : markerRadius);
    });
  }

  setSequence(poses) {
    this.profile = estimateHuman17Profile(poses);
    this._configureProfile();
    if (this.currentPose) this.setPose(this.currentPose);
    return this.profile;
  }

  setPose(pose, render = true) {
    this.currentPose = pose;
    if (!isHuman17Pose(pose)) {
      this.regionGroup.visible = false;
      this.skeletonGroup.visible = false;
      if (render) this.render();
      return false;
    }
    for (let index = 0; index < 17; index += 1) this._setPoint(index, pose);
    this._applyOptions();
    this._updateRegionGeometry();
    this._updateSkeletonOverlay();
    if (render) this.render();
    return true;
  }

  setView(nextView = {}, render = true) {
    if (Number.isFinite(nextView.yaw)) this.view.yaw = nextView.yaw;
    if (Number.isFinite(nextView.pitch)) this.view.pitch = Math.max(-1.45, Math.min(1.45, nextView.pitch));
    if (Number.isFinite(nextView.zoom)) this.view.zoom = Math.max(0.45, Math.min(3.2, nextView.zoom));
    if (Number.isFinite(nextView.skeletonScale)) this.view.skeletonScale = Math.max(0.25, Math.min(8, nextView.skeletonScale));
    this._applyView();
    this._updateCamera();
    if (render) this.render();
  }

  setOptions(nextOptions = {}, render = true) {
    if (typeof nextOptions.regionsVisible === "boolean") this.options.regionsVisible = nextOptions.regionsVisible;
    if (typeof nextOptions.skeletonVisible === "boolean") this.options.skeletonVisible = nextOptions.skeletonVisible;
    if (Number.isFinite(nextOptions.thickness)) this.options.thickness = Math.max(0.65, Math.min(1.55, nextOptions.thickness));
    this._applyOptions();
    if (isHuman17Pose(this.currentPose)) {
      this._updateRegionGeometry();
      this._updateSkeletonOverlay();
    }
    if (render) this.render();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width || this.canvas.clientWidth || 1));
    const height = Math.max(1, Math.round(rect.height || this.canvas.clientHeight || 1));
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this._updateCamera();
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    const geometries = new Set();
    const materials = new Set();
    this.scene.traverse((object) => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) {
        const list = Array.isArray(object.material) ? object.material : [object.material];
        list.forEach((material) => materials.add(material));
      }
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    this.renderer.dispose();
  }
}

export function createHuman17RegionViewer(canvas, options) {
  return new Human17RegionViewer(canvas, options);
}
