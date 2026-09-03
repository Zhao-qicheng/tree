import * as THREE from "three";

import {
  DEFAULT_PALETTE,
  HUMAN17_BONES,
  SELECTED_EXTRA_BONES,
  SELECTED_EXTRA_INDICES,
  WHOLEBODY133_BONES,
  computeFootFrame,
  computeHandFrame,
  computeHeadFrame,
  computeTorsoFrame,
  corePoseFrom,
  estimateHuman17Profile,
  isExtendedPose,
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

function applyFrameToObject(object, origin, right, up, forward) {
  const matrix = new THREE.Matrix4();
  matrix.makeBasis(
    new THREE.Vector3(right[0], right[1], right[2]),
    new THREE.Vector3(forward[0], forward[1], forward[2]),
    new THREE.Vector3(up[0], up[1], up[2]),
  );
  object.position.set(origin[0], origin[1], origin[2]);
  object.quaternion.setFromRotationMatrix(matrix);
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

function overlayColor(index, full = false) {
  if (full) {
    if ([1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19].includes(index) || (index >= 91 && index < 112)) {
      return new THREE.Color(DEFAULT_PALETTE.left);
    }
    if ([2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22].includes(index) || index >= 112) {
      return new THREE.Color(DEFAULT_PALETTE.right);
    }
    return new THREE.Color(index >= 23 && index < 91 ? "#d087d1" : DEFAULT_PALETTE.center);
  }
  if ([17, 18, 19, 23, 24, 25, 26, 27, 33, 35].includes(index)) {
    return new THREE.Color(DEFAULT_PALETTE.left);
  }
  if ([20, 21, 22, 28, 29, 30, 31, 32, 34, 36].includes(index)) {
    return new THREE.Color(DEFAULT_PALETTE.right);
  }
  return new THREE.Color("#f4d35e");
}

class NodeOverlay {
  constructor(parent, pointIndices, pairs, options = {}) {
    this.group = new THREE.Group();
    this.pointIndices = pointIndices;
    this.pairs = pairs;
    this.full = options.full === true;
    const pointGeometry = new THREE.SphereGeometry(1, 8, 6);
    const pointMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      vertexColors: true,
      transparent: true,
      opacity: this.full ? 0.76 : 0.96,
      depthTest: false,
      depthWrite: false,
    });
    this.points = new THREE.InstancedMesh(pointGeometry, pointMaterial, pointIndices.length);
    this.points.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.points.renderOrder = this.full ? 12 : 13;
    pointIndices.forEach((pointIndex, instanceIndex) => {
      this.points.setColorAt(instanceIndex, overlayColor(pointIndex, this.full));
    });
    if (this.points.instanceColor) this.points.instanceColor.needsUpdate = true;

    const positions = new Float32Array(Math.max(1, pairs.length * 2 * 3));
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    lineGeometry.setDrawRange(0, 0);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: this.full ? 0x7b8794 : 0xf4d35e,
      transparent: true,
      opacity: this.full ? 0.42 : 0.78,
      depthTest: false,
      depthWrite: false,
    });
    this.lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    this.lines.renderOrder = this.full ? 12 : 13;
    this.group.add(this.lines, this.points);
    this.group.visible = false;
    parent.add(this.group);
  }

  update(pose, valid, scores, threshold, radius) {
    if (!Array.isArray(pose)) {
      this.group.visible = false;
      return 0;
    }
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    let visibleCount = 0;
    const pointValid = (index) => {
      if (!isFinitePoint(pose[index])) return false;
      if (Array.isArray(valid) && valid[index] === false) return false;
      if (Array.isArray(scores) && Number(scores[index]) < threshold) return false;
      return true;
    };
    this.pointIndices.forEach((pointIndex, instanceIndex) => {
      if (pointValid(pointIndex)) {
        const point = pose[pointIndex];
        position.set(Number(point[0]), Number(point[1]), Number(point[2]));
        scale.setScalar(radius);
        visibleCount += 1;
      } else {
        position.set(0, 0, 0);
        scale.setScalar(0);
      }
      matrix.compose(position, quaternion, scale);
      this.points.setMatrixAt(instanceIndex, matrix);
    });
    this.points.instanceMatrix.needsUpdate = true;

    const attribute = this.lines.geometry.attributes.position;
    let vertexCount = 0;
    this.pairs.forEach(([first, second]) => {
      if (!pointValid(first) || !pointValid(second)) return;
      const a = pose[first];
      const b = pose[second];
      attribute.setXYZ(vertexCount, Number(a[0]), Number(a[1]), Number(a[2]));
      attribute.setXYZ(vertexCount + 1, Number(b[0]), Number(b[1]), Number(b[2]));
      vertexCount += 2;
    });
    attribute.needsUpdate = true;
    this.lines.geometry.setDrawRange(0, vertexCount);
    this.group.visible = true;
    return visibleCount;
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
      extendedVisible: options.extendedVisible !== false,
      feetVisible: options.feetVisible !== false,
      handsVisible: options.handsVisible !== false,
      faceMarkersVisible: options.faceMarkersVisible !== false,
      nodeMode: ["hidden", "selected", "full"].includes(options.nodeMode) ? options.nodeMode : "hidden",
      nodeThreshold: Number.isFinite(options.nodeThreshold) ? Number(options.nodeThreshold) : 0.25,
    };
    this.profile = estimateHuman17Profile([]);
    this.currentPose = null;
    this.extendedPose = null;
    this.extendedValid = null;
    this.extendedScores = null;
    this.wholebodyPose = null;
    this.wholebodyValid = null;
    this.wholebodyScores = null;
    this.visibleNodeCount = 0;
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
    this.nodeOverlayGroup = new THREE.Group();
    this.pitchGroup.add(this.yawGroup);
    this.yawGroup.add(this.referenceGroup, this.poseGroup);
    this.poseGroup.add(this.regionGroup, this.skeletonGroup, this.nodeOverlayGroup);
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
    this.selectedNodeOverlay = new NodeOverlay(
      this.nodeOverlayGroup,
      SELECTED_EXTRA_INDICES,
      SELECTED_EXTRA_BONES,
    );
    this.fullNodeOverlay = new NodeOverlay(
      this.nodeOverlayGroup,
      Array.from({ length: 133 }, (_, index) => index),
      WHOLEBODY133_BONES,
      { full: true },
    );
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
    this._buildOrientationParts();
  }

  _buildOrientationParts() {
    const visorMaterial = new THREE.MeshStandardMaterial({
      color: "#f4d35e",
      roughness: 0.42,
      metalness: 0.08,
    });
    const chestMaterial = new THREE.MeshStandardMaterial({
      color: "#f8f1d0",
      roughness: 0.48,
      metalness: 0.04,
    });
    this.materials.visor = visorMaterial;
    this.materials.chest = chestMaterial;

    this.chestMarker = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 3), chestMaterial);
    this.chestMarker.rotation.x = Math.PI / 2;
    this.regionGroup.add(this.chestMarker);

    this.faceVisor = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 4), visorMaterial);
    this.faceVisor.rotation.x = Math.PI / 2;
    this.regionGroup.add(this.faceVisor);

    this.orientedFeet = {
      left: this._createFootGroup(this.materials.left),
      right: this._createFootGroup(this.materials.right),
    };
    this.hands = {
      left: this._createHandGroup(this.materials.left),
      right: this._createHandGroup(this.materials.right),
    };
    this.fingerCapsules = {
      left: Array.from({ length: 5 }, () => new CapsulePart(this.regionGroup, this.materials.left, 10)),
      right: Array.from({ length: 5 }, () => new CapsulePart(this.regionGroup, this.materials.right, 10)),
    };
  }

  _createFootGroup(material) {
    const group = new THREE.Group();
    const foot = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    group.add(foot);
    group.userData = { foot };
    this.regionGroup.add(group);
    return group;
  }

  _createHandGroup(material) {
    const group = new THREE.Group();
    const palm = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    group.add(palm);
    group.userData = { palm };
    this.regionGroup.add(group);
    return group;
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
    this.nodeOverlayGroup.visible = this.options.nodeMode !== "hidden";
  }

  _updateNodeOverlays() {
    this.selectedNodeOverlay.group.visible = false;
    this.fullNodeOverlay.group.visible = false;
    this.visibleNodeCount = 0;
    const radius = this.profile.radii.joint
      * (this.options.nodeMode === "full" ? 0.58 : 0.78);
    if (this.options.nodeMode === "selected") {
      this.visibleNodeCount = this.selectedNodeOverlay.update(
        this.extendedPose,
        this.extendedValid,
        this.extendedScores,
        this.options.nodeThreshold,
        radius,
      );
      return this.visibleNodeCount;
    }
    if (this.options.nodeMode === "full") {
      this.visibleNodeCount = this.fullNodeOverlay.update(
        this.wholebodyPose,
        this.wholebodyValid,
        this.wholebodyScores,
        this.options.nodeThreshold,
        radius,
      );
    }
    return this.visibleNodeCount;
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
    this._updateExtendedGeometry();
  }

  _currentPoseArray() {
    if (this.extendedPose && isExtendedPose(this.extendedPose)) return this.extendedPose;
    if (!isHuman17Pose(this.currentPose)) return null;
    return this.currentPose;
  }

  _currentValidArray(pose) {
    if (this.extendedValid && this.extendedValid.length === pose.length) return this.extendedValid;
    return pose.map((point) => isFinitePoint(point));
  }

  _hideOrientationParts() {
    if (!this.chestMarker) return;
    this.chestMarker.visible = false;
    this.faceVisor.visible = false;
    this.orientedFeet.left.visible = false;
    this.orientedFeet.right.visible = false;
    this.hands.left.visible = false;
    this.hands.right.visible = false;
    this.fingerCapsules.left.forEach((part) => {
      part.group.visible = false;
    });
    this.fingerCapsules.right.forEach((part) => {
      part.group.visible = false;
    });
  }

  _updateExtendedGeometry() {
    if (!this.chestMarker || !this.options.regionsVisible) {
      this._hideOrientationParts();
      return;
    }
    const pose = this._currentPoseArray();
    if (!pose) {
      this._hideOrientationParts();
      return;
    }
    const valid = this._currentValidArray(pose);
    const torso = computeTorsoFrame(pose, valid);
    const showExtended = this.options.extendedVisible !== false;
    const thickness = this.options.thickness;
    const radii = this.profile.radii;

    if (torso && showExtended) {
      const size = radii.torsoTop * thickness * 1.15;
      applyFrameToObject(
        this.chestMarker,
        [
          torso.origin[0] + torso.forward[0] * size * 0.85,
          torso.origin[1] + torso.forward[1] * size * 0.85,
          torso.origin[2] + torso.forward[2] * size * 0.85,
        ],
        torso.right,
        torso.up,
        torso.forward,
      );
      this.chestMarker.scale.set(size * 0.85, size * 1.15, size * 0.35);
      this.chestMarker.visible = true;
    } else {
      this.chestMarker.visible = false;
    }

    const headFrame = computeHeadFrame(pose, valid, torso);
    if (headFrame && showExtended && this.options.faceMarkersVisible) {
      const size = radii.head * thickness * 0.42;
      applyFrameToObject(
        this.faceVisor,
        [
          headFrame.origin[0] + headFrame.forward[0] * size * 0.95,
          headFrame.origin[1] + headFrame.forward[1] * size * 0.95,
          headFrame.origin[2] + headFrame.forward[2] * size * 0.95,
        ],
        headFrame.right,
        headFrame.up,
        headFrame.forward,
      );
      this.faceVisor.scale.set(size * 0.9, size * 1.25, size * 0.45);
      this.faceVisor.visible = true;
    } else {
      this.faceVisor.visible = false;
    }

    this._updateOrientedFoot("left", pose, valid, torso, radii.foot * thickness);
    this._updateOrientedFoot("right", pose, valid, torso, radii.foot * thickness);
    this._updateOrientedHand("left", pose, valid, torso, radii.hand * thickness);
    this._updateOrientedHand("right", pose, valid, torso, radii.hand * thickness);
  }

  _updateOrientedFoot(side, pose, valid, torso, radius) {
    const group = this.orientedFeet[side];
    const frame = computeFootFrame(pose, valid, side, torso);
    const canShow = this.options.extendedVisible && this.options.feetVisible && frame;
    group.visible = Boolean(canShow);
    this.endpoints[`${side}Foot`].visible = this.endpoints[`${side}Foot`].visible && !canShow;
    if (!canShow) return;
    applyFrameToObject(group, frame.origin, frame.right, frame.up, frame.forward);
    const length = Math.max(frame.length, radius * 2.2);
    const width = Math.max(frame.width, radius * 1.3);
    const height = radius * 1.15;
    group.userData.foot.scale.set(width, length, height);
  }

  _updateOrientedHand(side, pose, valid, torso, radius) {
    const group = this.hands[side];
    const capsules = this.fingerCapsules[side];
    const frame = computeHandFrame(pose, valid, side, torso);
    const canShow = this.options.extendedVisible && this.options.handsVisible && frame;
    group.visible = Boolean(canShow);
    this.endpoints[`${side}Hand`].visible = this.endpoints[`${side}Hand`].visible && !canShow;
    if (!canShow) {
      capsules.forEach((part) => {
        part.group.visible = false;
      });
      return;
    }
    applyFrameToObject(group, frame.origin, frame.right, frame.up, frame.forward);
    group.userData.palm.scale.set(radius * 1.7, Math.max(frame.length * 0.42, radius * 1.4), radius * 0.55);
    const wrist = new THREE.Vector3(frame.wrist[0], frame.wrist[1], frame.wrist[2]);
    frame.tips.forEach((tip, index) => {
      if (index >= capsules.length) return;
      capsules[index].update(
        wrist,
        new THREE.Vector3(tip.point[0], tip.point[1], tip.point[2]),
        radius * 0.28,
      );
    });
    for (let index = frame.tips.length; index < capsules.length; index += 1) {
      capsules[index].group.visible = false;
    }
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
    if (this.extendedPose) this.setExtendedPose(this.extendedPose, this.extendedValid);
    else if (this.currentPose) this.setPose(this.currentPose);
    return this.profile;
  }

  setExtendedSequence(poses) {
    this.setSequence(poses);
    if (Array.isArray(poses) && poses.length && isExtendedPose(poses[0])) {
      this.setExtendedPose(poses[0], null, false);
    }
    return this.profile;
  }

  setPose(pose, render = true) {
    if (isExtendedPose(pose)) {
      return this.setExtendedPose(pose, null, render);
    }
    this.currentPose = pose;
    if (!isHuman17Pose(pose)) {
      this.regionGroup.visible = false;
      this.skeletonGroup.visible = false;
      this._hideOrientationParts();
      if (render) this.render();
      return false;
    }
    for (let index = 0; index < 17; index += 1) this._setPoint(index, pose);
    this._applyOptions();
    this._updateRegionGeometry();
    this._updateSkeletonOverlay();
    this._updateNodeOverlays();
    if (render) this.render();
    return true;
  }

  setExtendedPose(pose, valid = null, render = true) {
    if (!isExtendedPose(pose)) {
      this.extendedPose = null;
      this.extendedValid = null;
      return this.setPose(pose, render);
    }
    this.extendedPose = pose;
    this.extendedValid = Array.isArray(valid) ? valid : pose.map((point) => isFinitePoint(point));
    const ok = this.setPose(corePoseFrom(pose), false);
    if (render) this.render();
    return ok;
  }

  setNodeOverlayData(data = {}, render = true) {
    if (Array.isArray(data.selectedPose)) this.extendedPose = data.selectedPose;
    if (Array.isArray(data.selectedValid)) this.extendedValid = data.selectedValid;
    if (Array.isArray(data.selectedScores)) this.extendedScores = data.selectedScores;
    this.wholebodyPose = Array.isArray(data.wholebodyPose) ? data.wholebodyPose : null;
    this.wholebodyValid = Array.isArray(data.wholebodyValid) ? data.wholebodyValid : null;
    this.wholebodyScores = Array.isArray(data.wholebodyScores) ? data.wholebodyScores : null;
    const count = this._updateNodeOverlays();
    if (render) this.render();
    return count;
  }

  getVisibleNodeCount() {
    return this.visibleNodeCount;
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
    if (typeof nextOptions.extendedVisible === "boolean") this.options.extendedVisible = nextOptions.extendedVisible;
    if (typeof nextOptions.feetVisible === "boolean") this.options.feetVisible = nextOptions.feetVisible;
    if (typeof nextOptions.handsVisible === "boolean") this.options.handsVisible = nextOptions.handsVisible;
    if (typeof nextOptions.faceMarkersVisible === "boolean") this.options.faceMarkersVisible = nextOptions.faceMarkersVisible;
    if (["hidden", "selected", "full"].includes(nextOptions.nodeMode)) this.options.nodeMode = nextOptions.nodeMode;
    if (Number.isFinite(nextOptions.nodeThreshold)) {
      this.options.nodeThreshold = Math.max(0, Math.min(1, Number(nextOptions.nodeThreshold)));
    }
    if (Number.isFinite(nextOptions.thickness)) this.options.thickness = Math.max(0.65, Math.min(1.55, nextOptions.thickness));
    this._applyOptions();
    if (isHuman17Pose(this.currentPose)) {
      this._updateRegionGeometry();
      this._updateSkeletonOverlay();
    }
    this._updateNodeOverlays();
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
