(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin===`use-credentials`?t.credentials=`include`:e.crossOrigin===`anonymous`?t.credentials=`omit`:t.credentials=`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var e=`
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
`,t=`
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
`,n=`
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
`,r=`
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
`,i=33,a=2/(i-1),o=10,s=2e-5,c=4e-4,l=-9.8,u=.18,d=2.4,f=[.82,.83,.88],p=[.92,.12,.12],m=[.12,.35,.98],h=(e,t)=>t*i+e;function g(){let e=i*i,t=new Float32Array(e*4),n=new Float32Array(e*4);for(let e=0;e<i;e++)for(let r=0;r<i;r++){let i=h(r,e),o=-2/2+r*a,s=-2/2+e*a;t[i*4+0]=o,t[i*4+1]=0,t[i*4+2]=s,t[i*4+3]=1,n[i*4+0]=f[0],n[i*4+1]=f[1],n[i*4+2]=f[2],n[i*4+3]=1}let r=[h(0,0),h(i-1,0),h(0,i-1),h(i-1,i-1)];for(let e of r)t[e*4+3]=0,n[e*4+0]=p[0],n[e*4+1]=p[1],n[e*4+2]=p[2];let o=Math.floor(i/2),l=h(o,o);t[l*4+3]=0,n[l*4+0]=m[0],n[l*4+1]=m[1],n[l*4+2]=m[2];let u=[];for(let e=0;e<i-1;e++)for(let t=0;t<i-1;t++){let n=h(t,e),r=h(t+1,e),i=h(t,e+1),a=h(t+1,e+1);u.push(n,r,a),u.push(n,a,i)}let d=[[],[],[],[],[],[]];function g(e,n,r,i){let a=[t[n*4+0],t[n*4+1],t[n*4+2]],o=[t[r*4+0],t[r*4+1],t[r*4+2]],s=o[0]-a[0],c=o[1]-a[1],l=o[2]-a[2],u=Math.sqrt(s*s+c*c+l*l);d[e].push({p0:n,p1:r,restLength:u,compliance:i})}for(let e=0;e<i;e++)for(let t=0;t<i-1;t++)g(t%2,h(t,e),h(t+1,e),s);for(let e=0;e<i-1;e++)for(let t=0;t<i;t++)g(2+e%2,h(t,e),h(t,e+1),s);for(let e=0;e<i-1;e++)for(let t=0;t<i-1;t++)g(4+t%2,h(t,e),h(t+1,e+1),c);return{numVertices:e,positions:t,colors:n,indices:u,groups:d,drivenIndex:l,ci:o}}function _(e,t,n,r){let i=1/Math.tan(e/2),a=1/(n-r),o=new Float32Array(16);return o[0]=i/t,o[5]=i,o[10]=(r+n)*a,o[11]=-1,o[14]=2*r*n*a,o}function ee(e,t,n){let r=S(y(e,t)),i=S(b(n,r)),a=b(r,i),o=new Float32Array(16);return o[0]=i[0],o[1]=a[0],o[2]=r[0],o[3]=0,o[4]=i[1],o[5]=a[1],o[6]=r[1],o[7]=0,o[8]=i[2],o[9]=a[2],o[10]=r[2],o[11]=0,o[12]=-x(i,e),o[13]=-x(a,e),o[14]=-x(r,e),o[15]=1,o}function v(e,t){let n=new Float32Array(16);for(let r=0;r<4;r++)for(let i=0;i<4;i++)n[r*4+i]=e[0+i]*t[r*4+0]+e[4+i]*t[r*4+1]+e[8+i]*t[r*4+2]+e[12+i]*t[r*4+3];return n}function y(e,t){return[e[0]-t[0],e[1]-t[1],e[2]-t[2]]}function b(e,t){return[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]]}function x(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]}function S(e){let t=Math.sqrt(x(e,e))||1;return[e[0]/t,e[1]/t,e[2]/t]}async function C(){let i=document.getElementById(`gpuCanvas`),a=document.getElementById(`status`),s=document.getElementById(`gravityCheckbox`);if(!navigator.gpu){a.textContent=`WebGPU не поддерживается этим браузером. Откройте страницу в свежем Chrome/Edge.`;return}let c=await navigator.gpu.requestAdapter();if(!c){a.textContent=`Не удалось получить GPU-адаптер.`;return}let f=await c.requestDevice(),p=i.getContext(`webgpu`),m=navigator.gpu.getPreferredCanvasFormat();function h(){let e=Math.min(window.devicePixelRatio||1,2);i.width=Math.max(1,Math.floor(i.clientWidth*e)),i.height=Math.max(1,Math.floor(i.clientHeight*e))}h(),window.addEventListener(`resize`,()=>{h(),B()}),p.configure({device:f,format:m,alphaMode:`opaque`});let y=g(),b=y.numVertices,x=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,S=f.createBuffer({size:y.positions.byteLength,usage:x});f.queue.writeBuffer(S,0,y.positions);let C=f.createBuffer({size:y.positions.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});f.queue.writeBuffer(C,0,new Float32Array(b*4));let w=f.createBuffer({size:y.positions.byteLength,usage:x});f.queue.writeBuffer(w,0,y.positions);let T=f.createBuffer({size:y.colors.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});f.queue.writeBuffer(T,0,y.colors);let E=new Uint32Array(y.indices),D=f.createBuffer({size:E.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST});f.queue.writeBuffer(D,0,E);let O=y.groups.map(e=>{let t=e.length,n=new ArrayBuffer(t*16),r=new Uint32Array(n),i=new Float32Array(n);for(let n=0;n<t;n++)r[n*4+0]=e[n].p0,r[n*4+1]=e[n].p1,i[n*4+2]=e[n].restLength,i[n*4+3]=0;let a=f.createBuffer({size:Math.max(16,n.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});f.queue.writeBuffer(a,0,n);let o=f.createBuffer({size:Math.max(16,t*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return f.queue.writeBuffer(o,0,new Float32Array(Math.max(4,t))),{count:t,constraintBuf:a,lambdaBuf:o,compliance:e.length>0?e[0].compliance:0}}),k=new Float32Array(Math.max(1,...O.map(e=>e.count))),A=f.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),j=O.map(()=>f.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),M=f.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),N=f.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),P=f.createShaderModule({code:e}),te=f.createShaderModule({code:t}),ne=f.createShaderModule({code:n}),F=f.createShaderModule({code:r}),I=f.createComputePipeline({layout:`auto`,compute:{module:P,entryPoint:`main`}}),L=f.createComputePipeline({layout:`auto`,compute:{module:te,entryPoint:`main`}}),R=f.createComputePipeline({layout:`auto`,compute:{module:ne,entryPoint:`main`}}),z=null;function B(){z&&z.destroy(),z=f.createTexture({size:[i.width,i.height],format:`depth24plus`,usage:GPUTextureUsage.RENDER_ATTACHMENT})}B();let V=f.createRenderPipeline({layout:`auto`,vertex:{module:F,entryPoint:`vs_main`},fragment:{module:F,entryPoint:`fs_main`,targets:[{format:m}]},primitive:{topology:`triangle-list`,cullMode:`none`},depthStencil:{format:`depth24plus`,depthWriteEnabled:!0,depthCompare:`less`}}),H=f.createBindGroup({layout:I.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:S}},{binding:1,resource:{buffer:C}},{binding:2,resource:{buffer:w}},{binding:3,resource:{buffer:A}}]}),re=O.map((e,t)=>f.createBindGroup({layout:L.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:e.constraintBuf}},{binding:2,resource:{buffer:e.lambdaBuf}},{binding:3,resource:{buffer:j[t]}}]})),ie=f.createBindGroup({layout:R.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:S}},{binding:1,resource:{buffer:C}},{binding:2,resource:{buffer:w}},{binding:3,resource:{buffer:M}}]}),ae=f.createBindGroup({layout:V.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:N}},{binding:1,resource:{buffer:S}},{binding:2,resource:{buffer:T}}]});for(let e=0;e<O.length;e++)f.queue.writeBuffer(j[e],0,new Float32Array([0,O[e].compliance])),f.queue.writeBuffer(j[e],8,new Uint32Array([O[e].count,0]));let U=.65,W=.55,G=3.2,K=!1,q=0,J=0;i.addEventListener(`pointerdown`,e=>{K=!0,q=e.clientX,J=e.clientY}),window.addEventListener(`pointerup`,()=>{K=!1}),window.addEventListener(`pointermove`,e=>{if(!K)return;let t=e.clientX-q,n=e.clientY-J;q=e.clientX,J=e.clientY,U-=t*.006,W=Math.max(.08,Math.min(1.5,W-n*.006))}),i.addEventListener(`wheel`,e=>{e.preventDefault(),G=Math.max(1,Math.min(8,G+e.deltaY*.002))},{passive:!1});function oe(){let e=ee([G*Math.cos(W)*Math.sin(U),G*Math.sin(W),G*Math.cos(W)*Math.cos(U)],[0,0,0],[0,1,0]);return v(_(Math.PI/4,i.width/i.height,.05,50),e)}let Y=y.drivenIndex,se=y.positions[Y*4+0],ce=y.positions[Y*4+2],X=+!!s.checked;s.addEventListener(`change`,()=>{X=+!!s.checked});let Z=0,Q=performance.now();a.textContent=`Вершин: ${b}, треугольников: ${y.indices.length/3}.`;function $(){let e=performance.now(),t=(e-Q)/1e3;Q=e,t=Math.min(t,1/30);let n=t/o;for(let e=0;e<o;e++){Z+=n;for(let e of O)e.count>0&&f.queue.writeBuffer(e.lambdaBuf,0,k,0,e.count);f.queue.writeBuffer(A,0,new Float32Array([n,X,l,0])),f.queue.writeBuffer(A,12,new Uint32Array([Y])),f.queue.writeBuffer(A,16,new Float32Array([se,0,ce,Z,u,d])),f.queue.writeBuffer(A,40,new Uint32Array([b,0]));for(let e=0;e<O.length;e++){let t=O[e];t.count!==0&&f.queue.writeBuffer(j[e],0,new Float32Array([n,t.compliance]))}f.queue.writeBuffer(M,0,new Float32Array([n])),f.queue.writeBuffer(M,4,new Uint32Array([b]));let e=f.createCommandEncoder(),t=e.beginComputePass();t.setPipeline(I),t.setBindGroup(0,H),t.dispatchWorkgroups(Math.ceil(b/64)),t.setPipeline(L);for(let e=0;e<O.length;e++){let n=O[e];n.count!==0&&(t.setBindGroup(0,re[e]),t.dispatchWorkgroups(Math.ceil(n.count/64)))}t.setPipeline(R),t.setBindGroup(0,ie),t.dispatchWorkgroups(Math.ceil(b/64)),t.end(),f.queue.submit([e.finish()])}let r=oe();f.queue.writeBuffer(N,0,r);let i=f.createCommandEncoder(),a=p.getCurrentTexture().createView(),s=i.beginRenderPass({colorAttachments:[{view:a,clearValue:{r:.06,g:.07,b:.09,a:1},loadOp:`clear`,storeOp:`store`}],depthStencilAttachment:{view:z.createView(),depthClearValue:1,depthLoadOp:`clear`,depthStoreOp:`store`}});s.setPipeline(V),s.setBindGroup(0,ae),s.setIndexBuffer(D,`uint32`),s.drawIndexed(y.indices.length),s.end(),f.queue.submit([i.finish()]),requestAnimationFrame($)}requestAnimationFrame($)}C().catch(e=>{console.error(e);let t=document.getElementById(`status`);t&&(t.textContent=`Ошибка: `+e.message)});