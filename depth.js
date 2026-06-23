/* Depth Anything V2 Small — browser-side ONNX (main thread) */

const INPUT_SIZE = 518;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const INFERNO_ANCHORS = [
  [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
  [186, 54, 85], [227, 89, 51], [249, 140, 10], [252, 255, 164]
];

let infernoLUT = null;
let session = null;

function buildInfernoLUT() {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (INFERNO_ANCHORS.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;
    const a = INFERNO_ANCHORS[Math.min(idx, INFERNO_ANCHORS.length - 1)];
    const b = INFERNO_ANCHORS[Math.min(idx + 1, INFERNO_ANCHORS.length - 1)];
    lut[i * 3] = a[0] + (b[0] - a[0]) * frac;
    lut[i * 3 + 1] = a[1] + (b[1] - a[1]) * frac;
    lut[i * 3 + 2] = a[2] + (b[2] - a[2]) * frac;
  }
  return lut;
}

function configureOrt(wasmBase) {
  ort.env.wasm.wasmPaths = wasmBase;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
}

async function verifyWasmRuntime(wasmBase, onStatus) {
  onStatus?.('检查 WASM 运行时…');
  const wasmUrl = new URL('ort-wasm-simd-threaded.wasm', wasmBase).href;
  const resp = await fetch(wasmUrl);
  if (!resp.ok) {
    throw new Error(`WASM 文件不可访问 (${resp.status}): ${wasmUrl}`);
  }

  const contentType = (resp.headers.get('content-type') || '').toLowerCase();
  if (!contentType.includes('wasm')) {
    throw new Error(
      `WASM MIME 类型不正确（${contentType || 'unknown'}）。请运行 python serve.py，不要用 python -m http.server`
    );
  }
}

async function fetchModelBuffer(modelUrl, onProgress) {
  const resp = await fetch(modelUrl);
  if (!resp.ok) {
    throw new Error(`模型下载失败 (${resp.status}): ${modelUrl}`);
  }

  const total = Number(resp.headers.get('content-length') || 0);
  if (!resp.body) {
    const buffer = await resp.arrayBuffer();
    onProgress?.(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }

  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total || loaded);
  }

  const merged = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged.buffer;
}

async function createSessionWithProgress(modelBuffer, onStatus) {
  let seconds = 0;
  const timer = setInterval(() => {
    seconds += 1;
    onStatus?.(`解析 ONNX 模型… ${seconds}s（约 95MB，首次 10–60 秒属正常）`);
  }, 1000);

  try {
    return await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
      executionProviders: ['wasm'],
    });
  } finally {
    clearInterval(timer);
  }
}

async function initDepthModel({ modelUrl, wasmBase, onStatus }) {
  if (typeof ort === 'undefined') {
    throw new Error('ONNX Runtime 未加载：请确认 vendor/onnxruntime-web/ort.wasm.min.js 可访问');
  }
  if (typeof DepthEngine === 'undefined') {
    throw new Error('depth.js 未加载');
  }

  infernoLUT = buildInfernoLUT();
  configureOrt(wasmBase);
  await verifyWasmRuntime(wasmBase, onStatus);

  onStatus?.('下载模型文件…');
  const modelBuffer = await fetchModelBuffer(modelUrl, (loaded, total) => {
    if (total > 0) {
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      onStatus?.(
        `下载模型 ${pct}% (${(loaded / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(1)} MB)`
      );
    } else {
      onStatus?.(`下载模型 ${(loaded / 1024 / 1024).toFixed(1)} MB…`);
    }
  });

  session = await createSessionWithProgress(modelBuffer, onStatus);
  onStatus?.('模型就绪');
  return session;
}

function preprocessFromVideo(video) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) {
    throw new Error('视频尚未就绪');
  }

  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const tensor = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    tensor[i] = (r - MEAN[0]) / STD[0];
    tensor[plane + i] = (g - MEAN[1]) / STD[1];
    tensor[2 * plane + i] = (b - MEAN[2]) / STD[2];
  }
  return { tensor, width: w, height: h };
}

function depthToImageData(depthData, outWidth, outHeight) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < depthData.length; i++) {
    if (depthData[i] < min) min = depthData[i];
    if (depthData[i] > max) max = depthData[i];
  }
  const range = max - min + 1e-8;

  const small = document.createElement('canvas');
  small.width = INPUT_SIZE;
  small.height = INPUT_SIZE;
  const sctx = small.getContext('2d');
  const imageData = sctx.createImageData(INPUT_SIZE, INPUT_SIZE);
  const px = imageData.data;

  for (let i = 0; i < depthData.length; i++) {
    const v = Math.round(((depthData[i] - min) / range) * 255);
    const c = v * 3;
    px[i * 4] = infernoLUT[c];
    px[i * 4 + 1] = infernoLUT[c + 1];
    px[i * 4 + 2] = infernoLUT[c + 2];
    px[i * 4 + 3] = 255;
  }
  sctx.putImageData(imageData, 0, 0);

  const large = document.createElement('canvas');
  large.width = outWidth;
  large.height = outHeight;
  large.getContext('2d').drawImage(small, 0, 0, outWidth, outHeight);
  return large.getContext('2d').getImageData(0, 0, outWidth, outHeight);
}

async function inferDepthFromVideo(video) {
  if (!session) {
    throw new Error('模型尚未就绪');
  }

  const { tensor, width, height } = preprocessFromVideo(video);
  const inputName = session.inputNames[0];
  const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  const start = performance.now();
  const outputs = await session.run({ [inputName]: input });
  const elapsedMs = performance.now() - start;

  const depth = outputs[session.outputNames[0]].data;
  const imageData = depthToImageData(depth, width, height);
  return { imageData, elapsedMs, width, height };
}

window.DepthEngine = {
  initDepthModel,
  inferDepthFromVideo,
};
