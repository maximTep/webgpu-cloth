import { predictShader, solveShader, updateShader, renderShader } from './shaders.js';

// =============================================================================
// Параметры сетки ткани
// =============================================================================
const GRID_N = 33;           // вершин на сторону
const CLOTH_SIZE = 2.0;      // размер квадрата ткани в мировых единицах
const SPACING = CLOTH_SIZE / (GRID_N - 1);

const SUBSTEPS = 10;         // XPBD рекомендует много маленьких подшагов
const COMPLIANCE_STRUCT = 0.00002; // растяжимость структурных связей (почти жёсткие)
const COMPLIANCE_SHEAR = 0.0004;   // диагональные (сдвиговые) связи чуть мягче

// const GRAVITY_Y = -9.8;
const GRAVITY_Y = -9.8;
// const WAVE_AMPLITUDE = 0.18;
const WAVE_AMPLITUDE = 0.18;
const WAVE_FREQ = 2.4; // рад/с

const COLOR_BASE = [0.82, 0.83, 0.88];
const COLOR_PINNED = [0.92, 0.12, 0.12];
const COLOR_DRIVEN = [0.12, 0.35, 0.98];

const idx = (i, j) => j * GRID_N + i;

// =============================================================================
// Построение сетки: позиции, инвертированные массы, треугольники, constraints
// сгруппированные по "цветам" (graph coloring) для безопасного параллельного
// решения на GPU без атомарных операций.
// =============================================================================
function buildGrid() {
  const numVertices = GRID_N * GRID_N;
  const positions = new Float32Array(numVertices * 4); // x,y,z,invMass
  const colors = new Float32Array(numVertices * 4);

  for (let j = 0; j < GRID_N; j++) {
    for (let i = 0; i < GRID_N; i++) {
      const id = idx(i, j);
      const x = -CLOTH_SIZE / 2 + i * SPACING;
      const z = -CLOTH_SIZE / 2 + j * SPACING;
      positions[id * 4 + 0] = x;
      positions[id * 4 + 1] = 0;
      positions[id * 4 + 2] = z;
      positions[id * 4 + 3] = 1.0; // invMass по умолчанию

      colors[id * 4 + 0] = COLOR_BASE[0];
      colors[id * 4 + 1] = COLOR_BASE[1];
      colors[id * 4 + 2] = COLOR_BASE[2];
      colors[id * 4 + 3] = 1.0;
    }
  }

  // Угловые вершины — закреплены (пины), помечены красным.
  const corners = [idx(0, 0), idx(GRID_N - 1, 0), idx(0, GRID_N - 1), idx(GRID_N - 1, GRID_N - 1)];
  for (const c of corners) {
    positions[c * 4 + 3] = 0.0;
    colors[c * 4 + 0] = COLOR_PINNED[0];
    colors[c * 4 + 1] = COLOR_PINNED[1];
    colors[c * 4 + 2] = COLOR_PINNED[2];
  }

  // Центральная вершина — управляемая (кинематическая), помечена синим.
  const ci = Math.floor(GRID_N / 2);
  const drivenIndex = idx(ci, ci);
  positions[drivenIndex * 4 + 3] = 0.0;
  colors[drivenIndex * 4 + 0] = COLOR_DRIVEN[0];
  colors[drivenIndex * 4 + 1] = COLOR_DRIVEN[1];
  colors[drivenIndex * 4 + 2] = COLOR_DRIVEN[2];

  // ---- Треугольники (для рендера) ----
  // Каждая клетка (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1) режется одной диагональю
  // (i,j)-(i+1,j+1) на два треугольника, как показано на рисунке задания.
  const indices = [];
  for (let j = 0; j < GRID_N - 1; j++) {
    for (let i = 0; i < GRID_N - 1; i++) {
      const a = idx(i, j);
      const b = idx(i + 1, j);
      const c = idx(i, j + 1);
      const d = idx(i + 1, j + 1);
      indices.push(a, b, d);
      indices.push(a, d, c);
    }
  }

  // ---- Constraints (distance constraints) с раскраской по группам ----
  // Группы 0-1: горизонтальные рёбра, цвет = i % 2
  // Группы 2-3: вертикальные рёбра,   цвет = j % 2
  // Группы 4-5: диагональные (shear), цвет = i % 2 (вдоль диагональной цепочки)
  const groups = [[], [], [], [], [], []];

  function pushConstraint(groupId, p0, p1, compliance) {
    const p0v = [positions[p0 * 4 + 0], positions[p0 * 4 + 1], positions[p0 * 4 + 2]];
    const p1v = [positions[p1 * 4 + 0], positions[p1 * 4 + 1], positions[p1 * 4 + 2]];
    const dx = p1v[0] - p0v[0], dy = p1v[1] - p0v[1], dz = p1v[2] - p0v[2];
    const restLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
    groups[groupId].push({ p0, p1, restLength, compliance });
  }

  // Горизонтальные структурные связи
  for (let j = 0; j < GRID_N; j++) {
    for (let i = 0; i < GRID_N - 1; i++) {
      pushConstraint(i % 2, idx(i, j), idx(i + 1, j), COMPLIANCE_STRUCT);
    }
  }
  // Вертикальные структурные связи
  for (let j = 0; j < GRID_N - 1; j++) {
    for (let i = 0; i < GRID_N; i++) {
      pushConstraint(2 + (j % 2), idx(i, j), idx(i, j + 1), COMPLIANCE_STRUCT);
    }
  }
  // Диагональные (shear) связи — придают треугольникам сопротивление сдвигу
  for (let j = 0; j < GRID_N - 1; j++) {
    for (let i = 0; i < GRID_N - 1; i++) {
      pushConstraint(4 + (i % 2), idx(i, j), idx(i + 1, j + 1), COMPLIANCE_SHEAR);
    }
  }

  return { numVertices, positions, colors, indices, groups, drivenIndex, ci };
}

// =============================================================================
// Небольшая математика для камеры (без сторонних библиотек)
// =============================================================================
function mat4Perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function mat4LookAt(eye, center, up) {
  const z = normalize(sub(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -dot(x, eye); out[13] = -dot(y, eye); out[14] = -dot(z, eye); out[15] = 1;
  return out;
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function normalize(a) {
  const l = Math.sqrt(dot(a, a)) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

// =============================================================================
// Инициализация WebGPU и запуск
// =============================================================================
async function main() {
  const canvas = document.getElementById('gpuCanvas');
  const statusEl = document.getElementById('status');
  const gravityCheckbox = document.getElementById('gravityCheckbox');

  if (!navigator.gpu) {
    statusEl.textContent = 'WebGPU не поддерживается этим браузером. Откройте страницу в свежем Chrome/Edge.';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    statusEl.textContent = 'Не удалось получить GPU-адаптер.';
    return;
  }
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  }
  resize();
  window.addEventListener('resize', () => { resize(); createDepthTexture(); });

  context.configure({ device, format, alphaMode: 'opaque' });

  // ---- Данные сетки ----
  const grid = buildGrid();
  const numVertices = grid.numVertices;

  // ---- Буферы позиций/скоростей/предсказанных позиций ----
  const bufUsageStorage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const positionsBuf = device.createBuffer({
    size: grid.positions.byteLength, usage: bufUsageStorage,
  });
  device.queue.writeBuffer(positionsBuf, 0, grid.positions);

  const velocitiesBuf = device.createBuffer({
    size: grid.positions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(velocitiesBuf, 0, new Float32Array(numVertices * 4));

  const predictedBuf = device.createBuffer({
    size: grid.positions.byteLength, usage: bufUsageStorage,
  });
  device.queue.writeBuffer(predictedBuf, 0, grid.positions);

  const colorsBuf = device.createBuffer({
    size: grid.colors.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(colorsBuf, 0, grid.colors);

  // ---- Индексный буфер (треугольники) ----
  const indexArray = new Uint32Array(grid.indices);
  const indexBuf = device.createBuffer({
    size: indexArray.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuf, 0, indexArray);

  // ---- Буферы constraints и lambda по цветовым группам ----
  const colorGroups = grid.groups.map((constraints) => {
    const count = constraints.length;
    const data = new ArrayBuffer(count * 16); // {u32 p0, u32 p1, f32 restLength, f32 pad}
    const u32view = new Uint32Array(data);
    const f32view = new Float32Array(data);
    for (let k = 0; k < count; k++) {
      u32view[k * 4 + 0] = constraints[k].p0;
      u32view[k * 4 + 1] = constraints[k].p1;
      f32view[k * 4 + 2] = constraints[k].restLength;
      f32view[k * 4 + 3] = 0;
    }
    const constraintBuf = device.createBuffer({
      size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(constraintBuf, 0, data);

    const lambdaBuf = device.createBuffer({
      size: Math.max(16, count * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(lambdaBuf, 0, new Float32Array(Math.max(4, count)));

    // compliance одинаков внутри группы (структурные группы 0-3, shear 4-5)
    const compliance = constraints.length > 0 ? constraints[0].compliance : 0;

    return { count, constraintBuf, lambdaBuf, compliance };
  });
  const zeroLambdaScratch = new Float32Array(
    Math.max(1, ...colorGroups.map((g) => g.count))
  );

  // ---- Uniform-буферы ----
  const predictParamsBuf = device.createBuffer({
    size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const solveParamsBufs = colorGroups.map(() => device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  }));
  const updateParamsBuf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const renderUniformBuf = device.createBuffer({
    size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // =============================================================================
  // Пайплайны
  // =============================================================================
  const predictModule = device.createShaderModule({ code: predictShader });
  const solveModule = device.createShaderModule({ code: solveShader });
  const updateModule = device.createShaderModule({ code: updateShader });
  const renderModule = device.createShaderModule({ code: renderShader });

  const predictPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: predictModule, entryPoint: 'main' },
  });
  const solvePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: solveModule, entryPoint: 'main' },
  });
  const updatePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: updateModule, entryPoint: 'main' },
  });

  let depthTexture = null;
  function createDepthTexture() {
    if (depthTexture) depthTexture.destroy();
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  createDepthTexture();

  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  // ---- Bind groups ----
  const predictBindGroup = device.createBindGroup({
    layout: predictPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: positionsBuf } },
      { binding: 1, resource: { buffer: velocitiesBuf } },
      { binding: 2, resource: { buffer: predictedBuf } },
      { binding: 3, resource: { buffer: predictParamsBuf } },
    ],
  });

  const solveBindGroups = colorGroups.map((g, i) => device.createBindGroup({
    layout: solvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: predictedBuf } },
      { binding: 1, resource: { buffer: g.constraintBuf } },
      { binding: 2, resource: { buffer: g.lambdaBuf } },
      { binding: 3, resource: { buffer: solveParamsBufs[i] } },
    ],
  }));

  const updateBindGroup = device.createBindGroup({
    layout: updatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: positionsBuf } },
      { binding: 1, resource: { buffer: velocitiesBuf } },
      { binding: 2, resource: { buffer: predictedBuf } },
      { binding: 3, resource: { buffer: updateParamsBuf } },
    ],
  });

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUniformBuf } },
      { binding: 1, resource: { buffer: positionsBuf } },
      { binding: 2, resource: { buffer: colorsBuf } },
    ],
  });

  for (let i = 0; i < colorGroups.length; i++) {
    device.queue.writeBuffer(solveParamsBufs[i], 0, new Float32Array([0, colorGroups[i].compliance]));
    device.queue.writeBuffer(solveParamsBufs[i], 8, new Uint32Array([colorGroups[i].count, 0]));
  }

  // =============================================================================
  // Камера (орбита мышью + колесо зума)
  // =============================================================================
  let azimuth = 0.65, elevation = 0.55, radius = 3.2;
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('pointerup', () => { dragging = false; });
  window.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    azimuth -= dx * 0.006;
    elevation = Math.max(0.08, Math.min(1.5, elevation - dy * 0.006));
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    radius = Math.max(1.0, Math.min(8.0, radius + e.deltaY * 0.002));
  }, { passive: false });

  function getViewProj() {
    const eye = [
      radius * Math.cos(elevation) * Math.sin(azimuth),
      radius * Math.sin(elevation),
      radius * Math.cos(elevation) * Math.cos(azimuth),
    ];
    const view = mat4LookAt(eye, [0, 0, 0], [0, 1, 0]);
    const proj = mat4Perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 50);
    return mat4Multiply(proj, view);
  }

  // =============================================================================
  // Главный цикл
  // =============================================================================
  const driven = grid.drivenIndex;
  const drivenBaseX = grid.positions[driven * 4 + 0];
  const drivenBaseY = 0;
  const drivenBaseZ = grid.positions[driven * 4 + 2];

  let gravityEnabled = gravityCheckbox.checked ? 1 : 0;
  gravityCheckbox.addEventListener('change', () => {
    gravityEnabled = gravityCheckbox.checked ? 1 : 0;
  });

  let simTime = 0;
  let lastFrameTime = performance.now();
  statusEl.textContent = `Вершин: ${numVertices}, треугольников: ${grid.indices.length / 3}.`;

  function frame() {
    const now = performance.now();
    let frameDt = (now - lastFrameTime) / 1000;
    lastFrameTime = now;
    frameDt = Math.min(frameDt, 1 / 30); // защита от рывков при потере фокуса

    const dtSub = frameDt / SUBSTEPS;

    // ВАЖНО: device.queue.writeBuffer() выполняется в порядке вызова сразу,
    // независимо от того, когда будет submit-нут command encoder. Поэтому
    // каждый подшаг оформляется СВОИМ encoder'ом и СВОИМ submit — иначе все
    // writeBuffer для 12 подшагов "укладываются" в очередь до того, как
    // выполнится хотя бы один compute pass, и параметры всех подшагов
    // схлопнутся в последние записанные значения.
    for (let s = 0; s < SUBSTEPS; s++) {
      simTime += dtSub;

      for (const g of colorGroups) {
        if (g.count > 0) device.queue.writeBuffer(g.lambdaBuf, 0, zeroLambdaScratch, 0, g.count);
      }

      device.queue.writeBuffer(predictParamsBuf, 0, new Float32Array([
        dtSub, gravityEnabled, GRAVITY_Y, 0,
      ]));
      device.queue.writeBuffer(predictParamsBuf, 12, new Uint32Array([driven]));
      device.queue.writeBuffer(predictParamsBuf, 16, new Float32Array([
        drivenBaseX, drivenBaseY, drivenBaseZ, simTime, WAVE_AMPLITUDE, WAVE_FREQ,
      ]));
      device.queue.writeBuffer(predictParamsBuf, 40, new Uint32Array([numVertices, 0]));

      for (let i = 0; i < colorGroups.length; i++) {
        const g = colorGroups[i];
        if (g.count === 0) continue;
        device.queue.writeBuffer(solveParamsBufs[i], 0, new Float32Array([dtSub, g.compliance]));
      }
      device.queue.writeBuffer(updateParamsBuf, 0, new Float32Array([dtSub]));
      device.queue.writeBuffer(updateParamsBuf, 4, new Uint32Array([numVertices]));

      const stepEncoder = device.createCommandEncoder();
      const pass = stepEncoder.beginComputePass();

      pass.setPipeline(predictPipeline);
      pass.setBindGroup(0, predictBindGroup);
      pass.dispatchWorkgroups(Math.ceil(numVertices / 64));

      pass.setPipeline(solvePipeline);
      for (let i = 0; i < colorGroups.length; i++) {
        const g = colorGroups[i];
        if (g.count === 0) continue;
        pass.setBindGroup(0, solveBindGroups[i]);
        pass.dispatchWorkgroups(Math.ceil(g.count / 64));
      }

      pass.setPipeline(updatePipeline);
      pass.setBindGroup(0, updateBindGroup);
      pass.dispatchWorkgroups(Math.ceil(numVertices / 64));

      pass.end();
      device.queue.submit([stepEncoder.finish()]);
    }

    // ---- Рендер ----
    const viewProj = getViewProj();
    device.queue.writeBuffer(renderUniformBuf, 0, viewProj);

    const encoder = device.createCommandEncoder();
    const view = context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view, clearValue: { r: 0.06, g: 0.07, b: 0.09, a: 1 }, loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.setIndexBuffer(indexBuf, 'uint32');
    renderPass.drawIndexed(grid.indices.length);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = 'Ошибка: ' + err.message;
});

// document.addEventListener('DOMContentLoaded', () => {
//   const modal = document.getElementById('myModal');
//   const closeBtn = document.querySelector('.close-btn');

//   // Показываем окно при загрузке
//   modal.style.display = 'flex';

//   // Функция закрытия
//   const closeModal = () => {
//     modal.style.display = 'none';
//   };

//   // Закрытие по крестику
//   closeBtn.addEventListener('click', closeModal);

//   // Закрытие по клику на темный фон
//   window.addEventListener('click', (event) => {
//     if (event.target === modal) {
//       closeModal();
//     }
//   });
// });
