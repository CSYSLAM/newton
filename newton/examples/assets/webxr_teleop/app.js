"use strict";

const PROTOCOL_VERSION = 1;
const MAX_BUFFERED_BYTES = 64 * 1024;

const canvas = document.querySelector("#xr-canvas");
const overlay = document.querySelector("#xr-overlay");
const panel = document.querySelector("#teleop-panel");
const togglePanelButton = document.querySelector("#toggle-panel");
const toggleViewModeButton = document.querySelector("#toggle-view-mode");
const enterButton = document.querySelector("#enter-vr");
const resetButton = document.querySelector("#reset-scene");
const secureStatus = document.querySelector("#secure-status");
const socketStatus = document.querySelector("#socket-status");
const xrStatus = document.querySelector("#xr-status");
const frameStatus = document.querySelector("#frame-status");
const geometryStatus = document.querySelector("#geometry-status");
const sceneTitle = document.querySelector("#scene-title");
const sceneLead = document.querySelector("#scene-lead");
const sceneControls = document.querySelector("#scene-controls");

let socket = null;
let reconnectTimer = null;
let session = null;
let referenceSpace = null;
let referenceSpaceName = null;
let gl = null;
let renderer = null;
let latestScene = null;
let sceneGeometry = null;
let geometryPromise = null;
let sequence = 0;
let sentFrames = 0;
let droppedFrames = 0;
let viewerAnchorMatrix = null;
let sceneFromNewton = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
let sceneDrawList = [];
let previewFrameRequested = false;
let lastPreviewTimeMs = 0;
let initialPlacementPending = true;
let previousBPressed = false;
let previousXPressed = false;
let previousXRFrameTimeMs = null;
let latestViewerPoseMatrix = null;
let viewYawRadians = 0;
let viewPitchRadians = 0;
let viewMode = "observer";
let appliedSceneKind = null;
let appliedDeformationFrame = null;
const MIXED_CUBE_SCENE_KIND = "soft-rigid-cubes-into-bag";
const TSHIRT_SCENE_KIND = "bimanual-fold-tshirt";
const OBSERVER_VIEW_MODE = "observer";
const FIRST_PERSON_VIEW_MODE = "robot-first-person";
// These scene-kind fallbacks also update clients attached to an already-running server.
const MIXED_CUBE_CAMERA_DOLLY_METERS = 1.8;
const TSHIRT_CAMERA_DOLLY_METERS = 1.8;
const TSHIRT_CAMERA_HEIGHT_METERS = 0.35;
const TSHIRT_CAMERA_PITCH_OFFSET_DEGREES = -8;
const THUMBSTICK_DEADZONE = 0.18;
const VIEW_YAW_SPEED_RADIANS_S = 1.5;
const VIEW_PITCH_SPEED_RADIANS_S = 1.0;
const VIEW_PITCH_LIMIT_RADIANS = Math.PI / 3;
const streamId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
const squeezeState = new WeakMap();
const selectState = new WeakMap();

const ROLE_LABELS = {
  robot: "W1",
  plug: "插头",
  socket: "插座",
  chair: "椅子",
  bolt: "螺栓",
  nut: "螺母",
  bag: "袋子",
  shirt: "T 恤",
  "soft-cube": "软方块",
  "rigid-cube": "硬方块",
};

function setPanelHidden(hidden) {
  panel.hidden = hidden;
  overlay.classList.toggle("panel-hidden", hidden);
  togglePanelButton.textContent = hidden ? "显示面板" : "隐藏面板";
  togglePanelButton.setAttribute("aria-expanded", String(!hidden));
}

function supportsFirstPerson(scene) {
  return Boolean(scene?.viewControls?.firstPersonEnabled && scene?.firstPersonCamera);
}

function updateViewModeButton() {
  const available = supportsFirstPerson(latestScene);
  const sceneKind = latestScene?.sceneInfo?.kind ?? latestScene?.sceneKind;
  const needsBackendReload = sceneKind === TSHIRT_SCENE_KIND && !available;
  toggleViewModeButton.hidden = !available && !needsBackendReload;
  toggleViewModeButton.disabled = !available;
  toggleViewModeButton.textContent = needsBackendReload
    ? "第一人称需重载 Newton 进程"
    : viewMode === FIRST_PERSON_VIEW_MODE
      ? "切换到桌面观察"
      : "切换到机器人第一人称";
  toggleViewModeButton.setAttribute("aria-pressed", String(viewMode === FIRST_PERSON_VIEW_MODE));
}

function setViewMode(nextMode, viewerPoseMatrix = latestViewerPoseMatrix) {
  const firstPerson = nextMode === FIRST_PERSON_VIEW_MODE && supportsFirstPerson(latestScene);
  viewMode = firstPerson ? FIRST_PERSON_VIEW_MODE : OBSERVER_VIEW_MODE;
  viewYawRadians = 0;
  viewPitchRadians = 0;
  if (viewerPoseMatrix) {
    viewerAnchorMatrix = new Float32Array(viewerPoseMatrix);
    initialPlacementPending = false;
  }
  updateViewModeButton();
  updateSceneFromNewton();
}

function toggleViewMode(viewerPoseMatrix = latestViewerPoseMatrix) {
  setViewMode(
    viewMode === FIRST_PERSON_VIEW_MODE ? OBSERVER_VIEW_MODE : FIRST_PERSON_VIEW_MODE,
    viewerPoseMatrix,
  );
}

function applySceneInfo(scene) {
  const info = scene?.sceneInfo;
  const kind = info?.kind ?? scene?.sceneKind ?? "unknown";
  if (!info || kind === appliedSceneKind) {
    return;
  }
  appliedSceneKind = kind;
  sceneTitle.textContent = info.title ?? "Newton 遥操作";
  sceneLead.textContent = info.description ?? "Newton 实时物理与 Quest WebXR 遥操作。";
  if (Array.isArray(info.controls)) {
    sceneControls.replaceChildren();
    for (const control of info.controls) {
      if (!Array.isArray(control) || control.length !== 2) {
        continue;
      }
      const item = document.createElement("div");
      const key = document.createElement("kbd");
      const description = document.createElement("span");
      key.textContent = String(control[0]);
      description.textContent = String(control[1]);
      item.append(key, description);
      sceneControls.append(item);
    }
  }
}

function multiplyMat4(left, right) {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let index = 0; index < 4; index += 1) {
        value += left[index * 4 + row] * right[column * 4 + index];
      }
      output[column * 4 + row] = value;
    }
  }
  return output;
}

function rotateVector(quaternion, vector) {
  const [x, y, z, w] = quaternion;
  const [vx, vy, vz] = vector;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + y * tz - z * ty,
    vy + w * ty + z * tx - x * tz,
    vz + w * tz + x * ty - y * tx,
  ];
}

function modelMatrix(position, quaternion, scale) {
  const [x, y, z, w] = quaternion;
  const [sx, sy, sz] = scale;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  return new Float32Array([
    (1 - 2 * (yy + zz)) * sx,
    2 * (xy + wz) * sx,
    2 * (xz - wy) * sx,
    0,
    2 * (xy - wz) * sy,
    (1 - 2 * (xx + zz)) * sy,
    2 * (yz + wx) * sy,
    0,
    2 * (xz + wy) * sz,
    2 * (yz - wx) * sz,
    (1 - 2 * (xx + yy)) * sz,
    0,
    position[0],
    position[1],
    position[2],
    1,
  ]);
}

function normalizeVector(vector) {
  const length = Math.hypot(...vector);
  if (length < 1e-8) {
    throw new Error("相机方向向量无效");
  }
  return vector.map((value) => value / length);
}

function crossVector(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function rigidInverse(matrix) {
  const [tx, ty, tz] = [matrix[12], matrix[13], matrix[14]];
  return new Float32Array([
    matrix[0], matrix[4], matrix[8], 0,
    matrix[1], matrix[5], matrix[9], 0,
    matrix[2], matrix[6], matrix[10], 0,
    -(matrix[0] * tx + matrix[1] * ty + matrix[2] * tz),
    -(matrix[4] * tx + matrix[5] * ty + matrix[6] * tz),
    -(matrix[8] * tx + matrix[9] * ty + matrix[10] * tz),
    1,
  ]);
}

function poseFromMatrix(matrix) {
  const m00 = matrix[0];
  const m01 = matrix[4];
  const m02 = matrix[8];
  const m10 = matrix[1];
  const m11 = matrix[5];
  const m12 = matrix[9];
  const m20 = matrix[2];
  const m21 = matrix[6];
  const m22 = matrix[10];
  const trace = m00 + m11 + m22;
  let x;
  let y;
  let z;
  let w;
  if (trace > 0) {
    const scale = 2 * Math.sqrt(trace + 1);
    x = (m21 - m12) / scale;
    y = (m02 - m20) / scale;
    z = (m10 - m01) / scale;
    w = 0.25 * scale;
  } else if (m00 > m11 && m00 > m22) {
    const scale = 2 * Math.sqrt(1 + m00 - m11 - m22);
    x = 0.25 * scale;
    y = (m01 + m10) / scale;
    z = (m02 + m20) / scale;
    w = (m21 - m12) / scale;
  } else if (m11 > m22) {
    const scale = 2 * Math.sqrt(1 + m11 - m00 - m22);
    x = (m01 + m10) / scale;
    y = 0.25 * scale;
    z = (m12 + m21) / scale;
    w = (m02 - m20) / scale;
  } else {
    const scale = 2 * Math.sqrt(1 + m22 - m00 - m11);
    x = (m02 + m20) / scale;
    y = (m12 + m21) / scale;
    z = 0.25 * scale;
    w = (m10 - m01) / scale;
  }
  const length = Math.hypot(x, y, z, w);
  const sign = w < 0 ? -1 : 1;
  return {
    position: [matrix[12], matrix[13], matrix[14]],
    orientation: [x, y, z, w].map((value) => sign * value / length),
  };
}

function controllerPoseInNewton(pose) {
  if (!viewerAnchorMatrix || !latestScene?.camera) {
    return null;
  }
  const newtonFromReference = rigidInverse(sceneFromNewton);
  return poseFromMatrix(multiplyMat4(newtonFromReference, pose.transform.matrix));
}

function viewControlSettings(scene) {
  const configured = scene?.viewControls ?? {};
  const mixedCubeScene = scene?.sceneKind === MIXED_CUBE_SCENE_KIND;
  const legacyTshirtScene = scene?.sceneKind === TSHIRT_SCENE_KIND
    && configured.cameraHeightMeters === undefined
    && configured.cameraPitchOffsetDegrees === undefined;
  const dolly = Number(
    legacyTshirtScene
      ? TSHIRT_CAMERA_DOLLY_METERS
      : configured.cameraDollyMeters ?? (mixedCubeScene ? MIXED_CUBE_CAMERA_DOLLY_METERS : 0),
  );
  const height = Number(
    configured.cameraHeightMeters ?? (legacyTshirtScene ? TSHIRT_CAMERA_HEIGHT_METERS : 0),
  );
  const pitchOffset = Number(
    configured.cameraPitchOffsetDegrees
      ?? (legacyTshirtScene ? TSHIRT_CAMERA_PITCH_OFFSET_DEGREES : 0),
  );
  return {
    leftThumbstickRotate: Boolean(configured.leftThumbstickRotate ?? mixedCubeScene),
    cameraDollyMeters: Number.isFinite(dolly) ? Math.min(Math.max(dolly, 0), 5) : 0,
    cameraHeightMeters: Number.isFinite(height) ? Math.min(Math.max(height, -2), 2) : 0,
    cameraPitchOffsetDegrees: Number.isFinite(pitchOffset)
      ? Math.min(Math.max(pitchOffset, -45), 45)
      : 0,
  };
}

function viewRotationMatrix() {
  const yawHalfAngle = viewYawRadians * 0.5;
  const pitchHalfAngle = viewPitchRadians * 0.5;
  const yaw = modelMatrix(
    [0, 0, 0],
    [0, Math.sin(yawHalfAngle), 0, Math.cos(yawHalfAngle)],
    [1, 1, 1],
  );
  const pitch = modelMatrix(
    [0, 0, 0],
    [Math.sin(pitchHalfAngle), 0, 0, Math.cos(pitchHalfAngle)],
    [1, 1, 1],
  );
  return multiplyMat4(yaw, pitch);
}

function cameraWorldMatrix(camera, dollyMeters = 0, heightMeters = 0, pitchOffsetDegrees = 0) {
  const baseFront = normalizeVector(camera.front);
  const right = normalizeVector(crossVector(baseFront, camera.up));
  const baseUp = normalizeVector(crossVector(right, baseFront));
  const pitchRadians = pitchOffsetDegrees * Math.PI / 180;
  const pitchCosine = Math.cos(pitchRadians);
  const pitchSine = Math.sin(pitchRadians);
  const front = normalizeVector(
    baseFront.map((value, index) => value * pitchCosine + baseUp[index] * pitchSine),
  );
  const up = normalizeVector(
    baseUp.map((value, index) => value * pitchCosine - baseFront[index] * pitchSine),
  );
  const position = camera.position.map(
    (value, index) => value + baseFront[index] * dollyMeters + (index === 2 ? heightMeters : 0),
  );
  return new Float32Array([
    right[0], right[1], right[2], 0,
    up[0], up[1], up[2], 0,
    -front[0], -front[1], -front[2], 0,
    position[0], position[1], position[2], 1,
  ]);
}

function updateSceneFromNewton() {
  if (!viewerAnchorMatrix || !latestScene) {
    return;
  }
  const settings = viewControlSettings(latestScene);
  const firstPerson = viewMode === FIRST_PERSON_VIEW_MODE && supportsFirstPerson(latestScene);
  const camera = firstPerson ? latestScene.firstPersonCamera : latestScene.camera;
  if (!camera) {
    return;
  }
  const cameraMatrix = firstPerson
    ? cameraWorldMatrix(camera)
    : cameraWorldMatrix(
      camera,
      settings.cameraDollyMeters,
      settings.cameraHeightMeters,
      settings.cameraPitchOffsetDegrees,
    );
  sceneFromNewton = multiplyMat4(
    viewerAnchorMatrix,
    multiplyMat4(viewRotationMatrix(), rigidInverse(cameraMatrix)),
  );
}

function quaternionFromZ(direction) {
  const length = Math.hypot(...direction);
  if (length < 1e-7) {
    return [0, 0, 0, 1];
  }
  const dx = direction[0] / length;
  const dy = direction[1] / length;
  const dz = direction[2] / length;
  if (dz < -0.999999) {
    return [1, 0, 0, 0];
  }
  const quaternion = [-dy, dx, 0, 1 + dz];
  const norm = Math.hypot(...quaternion);
  return quaternion.map((value) => value / norm);
}

function compileShader(context, type, source) {
  const shader = context.createShader(type);
  context.shaderSource(shader, source);
  context.compileShader(shader);
  if (!context.getShaderParameter(shader, context.COMPILE_STATUS)) {
    throw new Error(context.getShaderInfoLog(shader) || "shader compilation failed");
  }
  return shader;
}

function createCubeVertices() {
  const values = [];
  const addFace = (a, b, c, d, normal) => {
    for (const vertex of [a, b, c, a, c, d]) {
      values.push(...vertex, ...normal);
    }
  };
  addFace([-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5], [0, 0, 1]);
  addFace([.5, -.5, -.5], [-.5, -.5, -.5], [-.5, .5, -.5], [.5, .5, -.5], [0, 0, -1]);
  addFace([.5, -.5, .5], [.5, -.5, -.5], [.5, .5, -.5], [.5, .5, .5], [1, 0, 0]);
  addFace([-.5, -.5, -.5], [-.5, -.5, .5], [-.5, .5, .5], [-.5, .5, -.5], [-1, 0, 0]);
  addFace([-.5, .5, .5], [.5, .5, .5], [.5, .5, -.5], [-.5, .5, -.5], [0, 1, 0]);
  addFace([-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5, .5], [-.5, -.5, .5], [0, -1, 0]);
  return new Float32Array(values);
}

function createRenderer(context) {
  const vertexShader = compileShader(context, context.VERTEX_SHADER, `
    attribute vec3 position;
    attribute vec3 normal;
    uniform mat4 model;
    uniform mat4 viewProjection;
    varying float light;
    void main() {
      vec3 worldNormal = normalize(mat3(model) * normal);
      light = 0.30 + 0.70 * max(dot(worldNormal, normalize(vec3(0.35, 0.85, 0.40))), 0.0);
      gl_Position = viewProjection * model * vec4(position, 1.0);
    }
  `);
  const fragmentShader = compileShader(context, context.FRAGMENT_SHADER, `
    precision mediump float;
    uniform vec3 color;
    varying float light;
    void main() {
      gl_FragColor = vec4(color * light, 1.0);
    }
  `);
  const program = context.createProgram();
  context.attachShader(program, vertexShader);
  context.attachShader(program, fragmentShader);
  context.linkProgram(program);
  if (!context.getProgramParameter(program, context.LINK_STATUS)) {
    throw new Error(context.getProgramInfoLog(program) || "shader link failed");
  }

  const positionLocation = context.getAttribLocation(program, "position");
  const normalLocation = context.getAttribLocation(program, "normal");
  const modelLocation = context.getUniformLocation(program, "model");
  const viewProjectionLocation = context.getUniformLocation(program, "viewProjection");
  const colorLocation = context.getUniformLocation(program, "color");
  const uint32Indices = context.getExtension("OES_element_index_uint");

  function createArrayGeometry(vertices) {
    const vertexBuffer = context.createBuffer();
    context.bindBuffer(context.ARRAY_BUFFER, vertexBuffer);
    context.bufferData(context.ARRAY_BUFFER, vertices, context.STATIC_DRAW);
    return { vertexBuffer, indexBuffer: null, indexType: null, count: vertices.length / 6 };
  }

  function createIndexedGeometry(interleaved, indices, vertexCount) {
    const vertexBuffer = context.createBuffer();
    context.bindBuffer(context.ARRAY_BUFFER, vertexBuffer);
    context.bufferData(context.ARRAY_BUFFER, interleaved, context.STATIC_DRAW);

    let indexData = indices;
    let indexType = context.UNSIGNED_INT;
    if (vertexCount <= 65535) {
      indexData = Uint16Array.from(indices);
      indexType = context.UNSIGNED_SHORT;
    } else if (!uint32Indices) {
      throw new Error("完整场景网格需要 OES_element_index_uint");
    }
    const indexBuffer = context.createBuffer();
    context.bindBuffer(context.ELEMENT_ARRAY_BUFFER, indexBuffer);
    context.bufferData(context.ELEMENT_ARRAY_BUFFER, indexData, context.STATIC_DRAW);
    return { vertexBuffer, indexBuffer, indexType, count: indices.length };
  }

  function bindGeometry(geometry) {
    context.bindBuffer(context.ARRAY_BUFFER, geometry.vertexBuffer);
    context.enableVertexAttribArray(positionLocation);
    context.vertexAttribPointer(positionLocation, 3, context.FLOAT, false, 24, 0);
    context.enableVertexAttribArray(normalLocation);
    context.vertexAttribPointer(normalLocation, 3, context.FLOAT, false, 24, 12);
    context.bindBuffer(context.ELEMENT_ARRAY_BUFFER, geometry.indexBuffer);
  }

  const cube = createArrayGeometry(createCubeVertices());
  let sceneMeshes = [];

  context.enable(context.DEPTH_TEST);
  context.enable(context.CULL_FACE);
  context.cullFace(context.BACK);
  let backfaceCullingEnabled = true;

  function setBackfaceCulling(enabled) {
    if (enabled === backfaceCullingEnabled) {
      return;
    }
    if (enabled) {
      context.enable(context.CULL_FACE);
    } else {
      context.disable(context.CULL_FACE);
    }
    backfaceCullingEnabled = enabled;
  }

  return {
    setSceneGeometry(geometry) {
      sceneMeshes = geometry.meshes.map((mesh) => (
        createIndexedGeometry(mesh.interleaved, mesh.indices, mesh.vertexCount)
      ));
    },
    updateSceneMesh(meshIndex, interleaved) {
      const geometry = sceneMeshes[meshIndex];
      if (!geometry) {
        return;
      }
      context.bindBuffer(context.ARRAY_BUFFER, geometry.vertexBuffer);
      context.bufferSubData(context.ARRAY_BUFFER, 0, interleaved);
    },
    begin(viewProjection) {
      context.useProgram(program);
      context.uniformMatrix4fv(viewProjectionLocation, false, viewProjection);
    },
    drawGeometry(matrix, color, geometry, doubleSided = false) {
      setBackfaceCulling(!doubleSided);
      bindGeometry(geometry);
      context.uniformMatrix4fv(modelLocation, false, matrix);
      context.uniform3fv(colorLocation, color);
      if (geometry.indexBuffer) {
        context.drawElements(context.TRIANGLES, geometry.count, geometry.indexType, 0);
      } else {
        context.drawArrays(context.TRIANGLES, 0, geometry.count);
      }
    },
    drawNewton(position, quaternion, scale, color) {
      const matrix = multiplyMat4(sceneFromNewton, modelMatrix(position, quaternion, scale));
      this.drawGeometry(matrix, color, cube);
    },
    drawSceneMesh(matrix, color, meshIndex, doubleSided = false) {
      const geometry = sceneMeshes[meshIndex];
      if (geometry) {
        this.drawGeometry(matrix, color, geometry, doubleSided);
      }
    },
  };
}

function parseSceneGeometry(buffer) {
  if (buffer.byteLength < 8) {
    throw new Error("场景网格数据过短");
  }
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== "NXR1") {
    throw new Error("场景网格格式不受支持");
  }
  const headerSize = new DataView(buffer).getUint32(4, true);
  const headerEnd = 8 + headerSize;
  const dataOffset = (headerEnd + 3) & ~3;
  if (dataOffset > buffer.byteLength) {
    throw new Error("场景网格头部不完整");
  }
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, headerEnd)));
  if (header.version !== 1 || header.vertexStrideFloats !== 6) {
    throw new Error(`场景网格版本不受支持：${header.version}`);
  }
  const meshes = header.meshes.map((mesh, meshIndex) => {
    const vertexOffset = dataOffset + mesh.vertexByteOffset;
    const indexOffset = dataOffset + mesh.indexByteOffset;
    const vertexFloatCount = mesh.vertexCount * header.vertexStrideFloats;
    if (
      vertexOffset % 4 !== 0
      || indexOffset % 4 !== 0
      || vertexOffset + vertexFloatCount * 4 > buffer.byteLength
      || indexOffset + mesh.indexCount * 4 > buffer.byteLength
    ) {
      throw new Error(`场景网格 ${meshIndex} 的缓冲区范围无效`);
    }
    return {
      interleaved: new Float32Array(buffer, vertexOffset, vertexFloatCount),
      indices: new Uint32Array(buffer, indexOffset, mesh.indexCount),
      vertexCount: mesh.vertexCount,
    };
  });
  const shapes = header.shapes.map((shape) => ({
    ...shape,
    localMatrix: modelMatrix(shape.position, shape.orientation, shape.scale),
  }));
  return { meshes, shapes, byteLength: buffer.byteLength };
}

async function loadSceneGeometry() {
  geometryStatus.textContent = "场景网格：下载中";
  const response = await fetch("/scene.bin", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`场景网格下载失败（HTTP ${response.status}）`);
  }
  const geometry = parseSceneGeometry(await response.arrayBuffer());
  sceneGeometry = geometry;
  appliedDeformationFrame = null;
  if (renderer) {
    renderer.setSceneGeometry(geometry);
  }
  const roleCounts = geometry.shapes.reduce((counts, shape) => {
    counts[shape.role] = (counts[shape.role] ?? 0) + 1;
    return counts;
  }, {});
  const detail = Object.entries(roleCounts)
    .map(([role, count]) => `${count} ${ROLE_LABELS[role] ?? role}`)
    .join(" + ");
  geometryStatus.textContent = `场景网格：${detail} · ${(
    geometry.byteLength / (1024 * 1024)
  ).toFixed(1)} MiB`;
  requestDesktopPreview();
  return geometry;
}

async function waitForSimulationReady() {
  for (;;) {
    try {
      const response = await fetch("/healthz", { cache: "no-store" });
      const health = await response.json();
      if (response.ok && health.simulationReady) {
        return;
      }
      frameStatus.textContent = "Newton 正在完成物理预热…";
    } catch (_error) {
      frameStatus.textContent = "正在等待 Newton 服务…";
    }
    await new Promise((resolve) => window.setTimeout(resolve, 250));
  }
}

function perspectiveMatrix(verticalFovDegrees, aspect, near = 0.01, far = 100.0) {
  const focalLength = 1 / Math.tan(verticalFovDegrees * Math.PI / 360);
  return new Float32Array([
    focalLength / aspect, 0, 0, 0,
    0, focalLength, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ]);
}

function ensureRenderer() {
  if (!gl) {
    gl = canvas.getContext("webgl", { alpha: false, antialias: true, depth: true, xrCompatible: true });
    if (!gl) {
      throw new Error("WebGL 不可用");
    }
  }
  if (!renderer) {
    renderer = createRenderer(gl);
    if (sceneGeometry) {
      renderer.setSceneGeometry(sceneGeometry);
    }
  }
}

function renderDesktopPreview(timeMs) {
  previewFrameRequested = false;
  if (session || !sceneGeometry || !latestScene) {
    return;
  }
  if (timeMs - lastPreviewTimeMs < 100) {
    window.setTimeout(requestDesktopPreview, 100 - (timeMs - lastPreviewTimeMs));
    return;
  }
  lastPreviewTimeMs = timeMs;
  ensureRenderer();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  viewerAnchorMatrix = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  updateSceneFromNewton();
  sceneDrawList = buildSceneDrawList(latestScene);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  gl.clearColor(0.025, 0.055, 0.09, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  renderer.begin(perspectiveMatrix(38, width / height));
  drawSimulationScene();
}

function requestDesktopPreview() {
  if (!session && !previewFrameRequested) {
    previewFrameRequested = true;
    window.requestAnimationFrame(renderDesktopPreview);
  }
}

function drawSegmentNewton(start, end, radius, color) {
  const direction = end.map((value, index) => value - start[index]);
  const length = Math.hypot(...direction);
  if (length < 1e-5) {
    return;
  }
  const midpoint = start.map((value, index) => (value + end[index]) * 0.5);
  renderer.drawNewton(midpoint, quaternionFromZ(direction), [radius * 2, radius * 2, length], color);
}

function drawStaticScene() {
  renderer.drawNewton([0, 0, -0.03], [0, 0, 0, 1], [4.0, 4.0, 0.06], [0.13, 0.17, 0.22]);
  for (const box of latestScene?.staticBoxes ?? []) {
    if (box.position?.length >= 3 && box.orientation?.length >= 4 && box.scale?.length >= 3) {
      renderer.drawNewton(box.position, box.orientation, box.scale, box.color ?? [0.35, 0.42, 0.48]);
    }
  }
  if (latestScene?.sceneKind === "plug-socket") {
    renderer.drawNewton([0.05, 0.13, 0.835], [0, 0, 0, 1], [0.56, 0.36, 0.02], [0.30, 0.39, 0.48]);
  }
}

function updateDeformableMeshes(scene) {
  const frameKey = `${scene.episode ?? 0}:${scene.frame ?? 0}`;
  if (!renderer || !sceneGeometry || appliedDeformationFrame === frameKey) {
    return;
  }
  for (const deformation of scene.deformableMeshes ?? []) {
    const mesh = sceneGeometry.meshes[deformation.mesh];
    const positions = deformation.positions;
    if (!mesh || !Array.isArray(positions) || positions.length !== mesh.vertexCount * 3) {
      continue;
    }
    const normals = new Float32Array(mesh.vertexCount * 3);
    for (let triangle = 0; triangle < mesh.indices.length; triangle += 3) {
      const ia = mesh.indices[triangle];
      const ib = mesh.indices[triangle + 1];
      const ic = mesh.indices[triangle + 2];
      const ax = positions[3 * ia];
      const ay = positions[3 * ia + 1];
      const az = positions[3 * ia + 2];
      const abx = positions[3 * ib] - ax;
      const aby = positions[3 * ib + 1] - ay;
      const abz = positions[3 * ib + 2] - az;
      const acx = positions[3 * ic] - ax;
      const acy = positions[3 * ic + 1] - ay;
      const acz = positions[3 * ic + 2] - az;
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      for (const index of [ia, ib, ic]) {
        normals[3 * index] += nx;
        normals[3 * index + 1] += ny;
        normals[3 * index + 2] += nz;
      }
    }
    for (let vertex = 0; vertex < mesh.vertexCount; vertex += 1) {
      const offset = 3 * vertex;
      const length = Math.hypot(normals[offset], normals[offset + 1], normals[offset + 2]);
      const inverseLength = length > 1e-8 ? 1 / length : 0;
      mesh.interleaved[6 * vertex] = positions[offset];
      mesh.interleaved[6 * vertex + 1] = positions[offset + 1];
      mesh.interleaved[6 * vertex + 2] = positions[offset + 2];
      mesh.interleaved[6 * vertex + 3] = length > 1e-8 ? normals[offset] * inverseLength : 0;
      mesh.interleaved[6 * vertex + 4] = length > 1e-8 ? normals[offset + 1] * inverseLength : 0;
      mesh.interleaved[6 * vertex + 5] = length > 1e-8 ? normals[offset + 2] * inverseLength : 1;
    }
    renderer.updateSceneMesh(deformation.mesh, mesh.interleaved);
  }
  appliedDeformationFrame = frameKey;
}

function drawTarget(target) {
  if (!target) {
    return;
  }
  const position = target.slice(0, 3);
  const quaternion = target.slice(3, 7);
  renderer.drawNewton(position, quaternion, [0.026, 0.026, 0.026], [1.0, 0.65, 0.12]);
  const axes = [
    [[0.07, 0, 0], [0.95, 0.18, 0.18]],
    [[0, 0.07, 0], [0.18, 0.95, 0.35]],
    [[0, 0, 0.07], [0.20, 0.50, 1.0]],
  ];
  for (const [localAxis, color] of axes) {
    const offset = rotateVector(quaternion, localAxis);
    const endpoint = position.map((value, index) => value + offset[index]);
    drawSegmentNewton(position, endpoint, 0.004, color);
  }
}

function buildSceneDrawList(scene) {
  if (!sceneGeometry) {
    return [];
  }
  updateDeformableMeshes(scene);
  const bodies = new Map();
  for (const body of scene.bodyPoses ?? scene.robotBodies ?? []) {
    if (body.length >= 8) {
      bodies.set(body[0], modelMatrix(body.slice(1, 4), body.slice(4, 8), [1, 1, 1]));
    }
  }
  for (const shape of sceneGeometry.shapes) {
    if (shape.role === "plug" && scene.plugPose?.length >= 7) {
      bodies.set(shape.body, modelMatrix(scene.plugPose.slice(0, 3), scene.plugPose.slice(3, 7), [1, 1, 1]));
    }
    if (shape.role === "chair" && scene.chairPose?.length >= 7) {
      bodies.set(shape.body, modelMatrix(scene.chairPose.slice(0, 3), scene.chairPose.slice(3, 7), [1, 1, 1]));
    }
  }
  const hiddenBodies = new Set(
    viewMode === FIRST_PERSON_VIEW_MODE ? scene.firstPersonHiddenBodies ?? [] : [],
  );
  const drawList = [];
  for (const shape of sceneGeometry.shapes) {
    if (hiddenBodies.has(shape.body)) {
      continue;
    }
    const bodyMatrix = bodies.get(shape.body);
    if (shape.body >= 0 && !bodyMatrix) {
      continue;
    }
    const worldMatrix = shape.body < 0
      ? shape.localMatrix
      : multiplyMat4(bodyMatrix, shape.localMatrix);
    drawList.push({
      matrix: multiplyMat4(sceneFromNewton, worldMatrix),
      color: shape.color,
      mesh: shape.mesh,
      // The role fallback keeps geometry packed before doubleSided was added usable.
      doubleSided: Boolean(shape.doubleSided || shape.role === "bag"),
    });
  }
  return drawList;
}

function drawSceneMeshes() {
  for (const shape of sceneDrawList) {
    renderer.drawSceneMesh(shape.matrix, shape.color, shape.mesh, shape.doubleSided);
  }
}

function drawSimulationScene() {
  drawStaticScene();
  if (!latestScene) {
    return;
  }
  drawSceneMeshes();
  const targets = latestScene.targetPoses ?? (latestScene.targetPose ? [latestScene.targetPose] : []);
  for (const target of targets) {
    drawTarget(target);
  }
}

function websocketUrl() {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
}

function connectSocket() {
  if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) {
    return;
  }
  socketStatus.textContent = "WebSocket：连接中";
  socket = new WebSocket(websocketUrl());
  socket.addEventListener("open", () => {
    socketStatus.textContent = "WebSocket：已连接";
    resetButton.disabled = false;
  });
  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "scene-state") {
        latestScene = message;
        applySceneInfo(message);
        if (viewMode === FIRST_PERSON_VIEW_MODE && !supportsFirstPerson(message)) {
          setViewMode(OBSERVER_VIEW_MODE);
        } else {
          updateViewModeButton();
        }
        updateSceneFromNewton();
        requestDesktopPreview();
        const mode = message.recording ? "录制中" : "已暂停";
        frameStatus.textContent = `Episode ${message.episode} · Newton #${message.frame} · ${mode} · ${message.recordedFrames} 帧`;
        resetButton.disabled = false;
      } else if (message.type === "reset-accepted") {
        frameStatus.textContent = `复位请求 #${message.resetRequest} 已送达 Newton`;
      } else if (message.type === "exit-immersive") {
        const activeSession = session;
        if (activeSession) {
          xrStatus.textContent = "WebXR：正在退出沉浸模式";
          activeSession.end().catch((error) => {
            xrStatus.textContent = `WebXR 退出失败：${error.message}`;
          });
        }
      } else if (message.type === "protocol-error") {
        socketStatus.textContent = `协议错误：${message.message}`;
      }
    } catch (error) {
      socketStatus.textContent = `场景数据错误：${error.message}`;
    }
  });
  socket.addEventListener("close", () => {
    socketStatus.textContent = "WebSocket：已断开，正在重连";
    resetButton.disabled = true;
    if (reconnectTimer === null) {
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connectSocket();
      }, 1000);
    }
  });
  socket.addEventListener("error", () => {
    socketStatus.textContent = "WebSocket：连接错误";
  });
}

function requestSceneReset() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    frameStatus.textContent = "无法复位：WebSocket 尚未连接";
    return;
  }
  resetButton.disabled = true;
  socket.send(JSON.stringify({
    type: "reset-scene",
    version: PROTOCOL_VERSION,
    requestId: `${streamId}-${Date.now()}`,
  }));
  frameStatus.textContent = "正在复位物理场景…";
}

function serializeController(frame, source) {
  if (!source.gripSpace || !["left", "right"].includes(source.handedness)) {
    return null;
  }
  const pose = frame.getPose(source.gripSpace, referenceSpace);
  if (!pose) {
    return null;
  }
  const newtonPose = controllerPoseInNewton(pose);
  if (!newtonPose) {
    return null;
  }
  const gamepad = source.gamepad;
  const axes = gamepad ? Array.from(gamepad.axes, Number) : [];
  return {
    pose: newtonPose,
    clutch: squeezeState.get(source) ?? Boolean(gamepad?.buttons?.[1]?.pressed),
    selecting: selectState.get(source) ?? Boolean(gamepad?.buttons?.[0]?.pressed),
    buttons: gamepad
      ? Array.from(gamepad.buttons, (button) => ({ pressed: button.pressed, value: Number(button.value) }))
      : [],
    axes,
    thumbstick: axes.length >= 2 ? axes.slice(-2) : [0, 0],
    triggerValue: Number(gamepad?.buttons?.[0]?.value ?? 0),
  };
}

function headPoseRelativeToAnchor(viewerPose) {
  if (!viewerAnchorMatrix) {
    return null;
  }
  return poseFromMatrix(
    multiplyMat4(rigidInverse(viewerAnchorMatrix), viewerPose.transform.matrix),
  );
}

function sendControllerFrame(frame, timeMs, viewerPose) {
  if (!socket || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > MAX_BUFFERED_BYTES) {
    droppedFrames += 1;
    return;
  }
  const controllers = {};
  for (const source of session.inputSources) {
    const controller = serializeController(frame, source);
    if (controller) {
      controllers[source.handedness] = controller;
    }
  }
  socket.send(JSON.stringify({
    type: "xr-frame",
    version: PROTOCOL_VERSION,
    streamId,
    sequence,
    timeMs,
    referenceSpace: referenceSpaceName,
    controllerSpace: "newton-world",
    visibilityState: session.visibilityState,
    viewMode,
    headPose: headPoseRelativeToAnchor(viewerPose),
    controllers,
  }));
  sequence += 1;
  sentFrames += 1;
}

function updateScenePlacement(viewerPose) {
  let bPressed = false;
  let xPressed = false;
  for (const source of session.inputSources) {
    if (source.handedness === "right") {
      bPressed = Boolean(source.gamepad?.buttons?.[5]?.pressed);
    } else if (source.handedness === "left") {
      xPressed = Boolean(source.gamepad?.buttons?.[4]?.pressed);
    }
  }
  if (xPressed && !previousXPressed && supportsFirstPerson(latestScene)) {
    toggleViewMode(viewerPose.transform.matrix);
  }
  if (initialPlacementPending || (bPressed && !previousBPressed)) {
    viewYawRadians = 0;
    viewPitchRadians = 0;
    viewerAnchorMatrix = new Float32Array(viewerPose.transform.matrix);
    initialPlacementPending = false;
    updateSceneFromNewton();
  }
  previousBPressed = bPressed;
  previousXPressed = xPressed;
}

function thumbstickValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude <= THUMBSTICK_DEADZONE) {
    return 0;
  }
  return Math.sign(value) * (magnitude - THUMBSTICK_DEADZONE) / (1 - THUMBSTICK_DEADZONE);
}

function updateViewRotation(timeMs) {
  const previousTimeMs = previousXRFrameTimeMs;
  previousXRFrameTimeMs = timeMs;
  if (
    previousTimeMs === null
    || viewMode === FIRST_PERSON_VIEW_MODE
    || !viewControlSettings(latestScene).leftThumbstickRotate
  ) {
    return;
  }
  let thumbstick = null;
  for (const source of session.inputSources) {
    if (source.handedness === "left" && source.gamepad?.axes?.length >= 2) {
      thumbstick = Array.from(source.gamepad.axes, Number).slice(-2);
      break;
    }
  }
  if (!thumbstick) {
    return;
  }
  const yawInput = thumbstickValue(thumbstick[0]);
  const pitchInput = thumbstickValue(thumbstick[1]);
  if (yawInput === 0 && pitchInput === 0) {
    return;
  }
  const deltaSeconds = Math.min(Math.max((timeMs - previousTimeMs) / 1000, 0), 0.05);
  viewYawRadians += yawInput * VIEW_YAW_SPEED_RADIANS_S * deltaSeconds;
  viewPitchRadians = Math.min(
    Math.max(viewPitchRadians + pitchInput * VIEW_PITCH_SPEED_RADIANS_S * deltaSeconds, -VIEW_PITCH_LIMIT_RADIANS),
    VIEW_PITCH_LIMIT_RADIANS,
  );
  updateSceneFromNewton();
}

function onXRFrame(timeMs, frame) {
  session.requestAnimationFrame(onXRFrame);
  const layer = session.renderState.baseLayer;
  const viewerPose = frame.getViewerPose(referenceSpace);
  if (!viewerPose) {
    return;
  }
  latestViewerPoseMatrix = new Float32Array(viewerPose.transform.matrix);
  updateScenePlacement(viewerPose);
  updateViewRotation(timeMs);
  sendControllerFrame(frame, timeMs, viewerPose);
  if (!layer || !renderer) {
    return;
  }
  sceneDrawList = latestScene ? buildSceneDrawList(latestScene) : [];
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0.025, 0.055, 0.09, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  for (const view of viewerPose.views) {
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    renderer.begin(multiplyMat4(view.projectionMatrix, view.transform.inverse.matrix));
    drawSimulationScene();
  }
}

async function enterVR() {
  try {
    connectSocket();
    const geometry = await geometryPromise;
    if (!geometry) {
      throw new Error("完整场景网格尚未加载");
    }
    ensureRenderer();
    await gl.makeXRCompatible();
    session = await navigator.xr.requestSession("immersive-vr", {
      optionalFeatures: ["local-floor", "bounded-floor", "dom-overlay"],
      domOverlay: { root: overlay },
    });
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    try {
      referenceSpaceName = "local-floor";
      referenceSpace = await session.requestReferenceSpace(referenceSpaceName);
    } catch (_error) {
      referenceSpaceName = "local";
      referenceSpace = await session.requestReferenceSpace(referenceSpaceName);
    }
    session.addEventListener("squeezestart", (event) => squeezeState.set(event.inputSource, true));
    session.addEventListener("squeezeend", (event) => squeezeState.set(event.inputSource, false));
    session.addEventListener("selectstart", (event) => selectState.set(event.inputSource, true));
    session.addEventListener("selectend", (event) => selectState.set(event.inputSource, false));
    session.addEventListener("end", () => {
      session = null;
      referenceSpace = null;
      referenceSpaceName = null;
      viewerAnchorMatrix = null;
      latestViewerPoseMatrix = null;
      sceneDrawList = [];
      previousXRFrameTimeMs = null;
      previousXPressed = false;
      viewYawRadians = 0;
      viewPitchRadians = 0;
      viewMode = OBSERVER_VIEW_MODE;
      updateViewModeButton();
      enterButton.disabled = false;
      xrStatus.textContent = "WebXR：会话已结束";
      requestDesktopPreview();
    });
    initialPlacementPending = true;
    previousXRFrameTimeMs = null;
    previousXPressed = false;
    viewYawRadians = 0;
    viewPitchRadians = 0;
    xrStatus.textContent = `WebXR：运行中（${referenceSpaceName}）`;
    enterButton.disabled = true;
    session.requestAnimationFrame(onXRFrame);
  } catch (error) {
    session = null;
    xrStatus.textContent = `WebXR 启动失败：${error.message}`;
    enterButton.disabled = false;
  }
}

async function initialize() {
  secureStatus.textContent = `安全上下文：${window.isSecureContext ? "是" : "否"}`;
  await waitForSimulationReady();
  connectSocket();
  geometryPromise = loadSceneGeometry().catch((error) => {
    geometryStatus.textContent = `场景网格：${error.message}`;
    return null;
  });
  if (!navigator.xr) {
    const geometry = await geometryPromise;
    if (geometry) {
      ensureRenderer();
      requestDesktopPreview();
    }
    xrStatus.textContent = "WebXR：当前浏览器不支持";
    return;
  }
  try {
    const supported = await navigator.xr.isSessionSupported("immersive-vr");
    const geometry = await geometryPromise;
    if (geometry) {
      ensureRenderer();
      requestDesktopPreview();
    }
    xrStatus.textContent = supported ? "WebXR：Quest 沉浸模式可用" : "WebXR：沉浸模式不可用";
    enterButton.disabled = !supported || !geometry;
  } catch (error) {
    xrStatus.textContent = `WebXR 检查失败：${error.message}`;
  }
}

enterButton.addEventListener("click", enterVR);
resetButton.addEventListener("click", requestSceneReset);
togglePanelButton.addEventListener("click", () => setPanelHidden(!panel.hidden));
toggleViewModeButton.addEventListener("click", () => toggleViewMode());
initialize();
