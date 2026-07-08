// ============================================================================
// WGSL-шейдеры для симуляции ткани методом XPBD (Position Based Dynamics
// с compliance, см. Macklin, Müller, Chentanez, "XPBD: Position-Based
// Simulation of Compliant Constrained Dynamics", 2016).
// ============================================================================

// ---- 1. PREDICT ------------------------------------------------------------
// Для каждой вершины делаем шаг явного интегрирования (semi-implicit Euler):
// v += g*dt; p_pred = p + v*dt. Закреплённые вершины (invMass == 0) остаются
// на месте. Управляемая ("ведущая") вершина двигается по закону синуса —
// именно ПЕРЕМЕЩАЕТСЯ (кинематически), а не получает силу.
export const predictShader = /* wgsl */ `
struct Params {
  dt: f32,
  gravityEnabled: f32,
  gravityY: f32,
  drivenIndex: u32,
  drivenBaseX: f32,
  drivenBaseY: f32,
  drivenBaseZ: f32,
  time: f32,
  amplitude: f32,
  angularFreq: f32,
  numVertices: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read>       positions : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> predicted : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params    : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numVertices) { return; }

  // Ведущая вершина: кинематическое перемещение по синусу по вертикали (Y).
  if (i == params.drivenIndex) {
    let y = params.drivenBaseY + params.amplitude * sin(params.angularFreq * params.time);
    predicted[i]  = vec4<f32>(params.drivenBaseX, y, params.drivenBaseZ, 0.0);
    velocities[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return;
  }

  var pos: vec4<f32> = positions[i];
  let invMass = pos.w;

  // Закреплённые (пиновые) вершины — не двигаются.
  if (invMass <= 0.0) {
    predicted[i] = vec4<f32>(pos.xyz, invMass);
    return;
  }

  var vel: vec4<f32> = velocities[i];
  if (params.gravityEnabled > 0.5) {
    vel.y = vel.y + params.gravityY * params.dt;
  }
  let newPos = pos.xyz + vel.xyz * params.dt;
  predicted[i] = vec4<f32>(newPos, invMass);
  velocities[i] = vel;
}
`;

// ---- 2. SOLVE CONSTRAINT (XPBD, distance constraint) -----------------------
// Ограничения разбиты на "цветовые" группы (graph coloring) так, что внутри
// одной группы ни одна вершина не встречается дважды. Это позволяет решать
// все constraints одной группы параллельно на GPU без атомарных операций и
// без гонок за память — потоки внутри одного dispatch пишут в непересекающиеся
// адреса storage-буфера.
export const solveShader = /* wgsl */ `
struct Constraint {
  p0: u32,
  p1: u32,
  restLength: f32,
  _pad: f32,
};

struct SolveParams {
  dt: f32,
  compliance: f32,
  numConstraints: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read_write> predicted   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       constraints : array<Constraint>;
@group(0) @binding(2) var<storage, read_write> lambdas     : array<f32>;
@group(0) @binding(3) var<uniform>             params      : SolveParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.numConstraints) { return; }

  let c = constraints[idx];
  var pv0 = predicted[c.p0];
  var pv1 = predicted[c.p1];
  let w0 = pv0.w;
  let w1 = pv1.w;
  let wsum = w0 + w1;
  if (wsum <= 0.0) { return; }

  let delta = pv1.xyz - pv0.xyz;
  let dist = length(delta);
  if (dist < 1e-8) { return; }
  let n = delta / dist;
  let C = dist - c.restLength;

  // XPBD: compliance alpha (обратная жёсткости). alpha=0 -> жёсткий constraint.
  let alphaTilde = params.compliance / (params.dt * params.dt);
  let denom = wsum + alphaTilde;
  let dLambda = (-C - alphaTilde * lambdas[idx]) / denom;
  lambdas[idx] = lambdas[idx] + dLambda;

  predicted[c.p0] = vec4<f32>(pv0.xyz - w0 * dLambda * n, w0);
  predicted[c.p1] = vec4<f32>(pv1.xyz + w1 * dLambda * n, w1);
}
`;

// ---- 3. UPDATE VELOCITY -----------------------------------------------------
// v = (p_pred - p) / dt ; p = p_pred
export const updateShader = /* wgsl */ `
struct Params2 {
  dt: f32,
  numVertices: u32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read_write> positions : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       predicted : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params    : Params2;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numVertices) { return; }
  let oldPos = positions[i];
  let newPos = predicted[i];
  if (params.dt > 0.0) {
    velocities[i] = vec4<f32>((newPos.xyz - oldPos.xyz) / params.dt, 0.0);
  }
  positions[i] = newPos;
}
`;

// ---- 4. RENDER --------------------------------------------------------------
// Позиции читаются прямо из storage-буфера по vertex_index (индексный буфер
// задаёт треугольники), поэтому отдельный vertex-buffer не нужен. Нормаль
// для простого освещения считается на лету через экранные производные
// (dpdx/dpdy) — без отдельного compute-прохода за нормалями.
export const renderShader = /* wgsl */ `
struct Uniforms {
  viewProj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> positions : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vertexColors : array<vec4<f32>>;

struct VSOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  let p = positions[vid].xyz;
  var out: VSOut;
  out.clipPos = uniforms.viewProj * vec4<f32>(p, 1.0);
  out.worldPos = p;
  out.color = vertexColors[vid].xyz;
  return out;
}

@fragment
fn fs_main(@location(0) worldPos: vec3<f32>, @location(1) color: vec3<f32>) -> @location(0) vec4<f32> {
  let dx = dpdx(worldPos);
  let dy = dpdy(worldPos);
  var normal = normalize(cross(dx, dy));
  // Освещаем обе стороны одинаково (двустороннее полотно ткани).
  let lightDir = normalize(vec3<f32>(0.4, 1.0, 0.5));
  let diff = abs(dot(normal, lightDir));
  let ambient = 0.35;
  let lit = color * (ambient + (1.0 - ambient) * diff);
  return vec4<f32>(lit, 1.0);
}
`;
