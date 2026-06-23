/* Depth Anything V2 — browser-side ONNX (main thread) */

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const DB_NAME = 'camera-depth-cache';
const DB_STORE = 'models';

const INFERNO_ANCHORS = [
  [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
  [186, 54, 85], [227, 89, 51], [249, 140, 10], [252, 255, 164]
];

let infernoLUT = null;
let session = null;
let inputSize = 518;
let modelCacheKey = 'depth_anything_v2_small_v1';
let forceSingleThread = false;

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

async function loadRuntimeConfig(configUrl = 'models/runtime_config.json') {
  try {
    const resp = await fetch(configUrl, { cache: 'no-store' });
    if (!resp.ok) {
      return null;
    }
    const cfg = await resp.json();
    inputSize = cfg.inputSize || inputSize;
    modelCacheKey = cfg.cacheKey || modelCacheKey;
    forceSingleThread = !!cfg.forceSingleThread;
    return cfg;
  } catch {
    return null;
  }
}

function configureOrt(wasmBase) {
  ort.env.wasm.wasmPaths = wasmBase;
  ort.env.wasm.simd = true;

  if (forceSingleThread) {
    ort.env.wasm.numThreads = 1;
    return 1;
  }

  const canMultiThread = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  const cores = navigator.hardwareConcurrency || 4;
  ort.env.wasm.numThreads = canMultiThread ? Math.min(4, cores) : 1;
  return ort.env.wasm.numThreads;
}

function formatProgress(loaded, total) {
  const loadedMb = (loaded / 1024 / 1024).toFixed(1);
  if (total > 0) {
    const pct = Math.min(100, Math.round((loaded / total) * 100));
    const totalMb = (total / 1024 / 1024).toFixed(1);
    return `下载模型 ${pct}% (${loadedMb} / ${totalMb} MB)`;
  }
  return `下载模型 ${loadedMb} MB…`;
}

function openModelDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(DB_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB 打开失败'));
  });
}

async function loadCachedModel() {
  try {
    const db = await openModelDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readonly');
      const get = tx.objectStore(DB_STORE).get(modelCacheKey);
      get.onsuccess = () => resolve(get.result || null);
      get.onerror = () => reject(get.error);
    });
  } catch {
    return null;
  }
}

async function saveCachedModel(buffer) {
  try {
    const db = await openModelDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readwrite');
      tx.objectStore(DB_STORE).put(buffer, modelCacheKey);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.warn('模型缓存写入失败（不影响使用）:', err);
  }
}

async function verifyWasmRuntime(wasmBase, onStatus) {
  onStatus?.('检查 WASM 运行时…');
  const wasmUrl = new URL('ort-wasm-simd-threaded.wasm', wasmBase).href;
  const resp = await fetch(wasmUrl, { method: 'HEAD' });
  if (!resp.ok) {
    throw new Error(`WASM 文件不可访问 (${resp.status}): ${wasmUrl}`);
  }

  const contentType = (resp.headers.get('content-type') || '').toLowerCase();
  if (!contentType.includes('wasm')) {
    throw new Error(
      `WASM MIME 类型不正确（${contentType || 'unknown'}）。请运行 python serve.py`
    );
  }
}

function downloadArrayBuffer(url, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.timeout = 10 * 60 * 1000;

    let lastLoaded = 0;
    let stallTimer = null;

    const resetStallTimer = () => {
      if (stallTimer) clearInterval(stallTimer);
      stallTimer = setInterval(() => {
        onProgress?.(lastLoaded, xhr.total || 0, true);
      }, 3000);
    };

    xhr.onprogress = (event) => {
      lastLoaded = event.loaded;
      const total = event.lengthComputable ? event.total : xhr.total || 0;
      onProgress?.(event.loaded, total, false);
      resetStallTimer();
    };

    xhr.onload = () => {
      if (stallTimer) clearInterval(stallTimer);
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
        resolve(xhr.response);
        return;
      }
      reject(new Error(`模型下载失败 (HTTP ${xhr.status})`));
    };

    xhr.onerror = () => {
      if (stallTimer) clearInterval(stallTimer);
      reject(new Error('网络错误：请确认手机与电脑在同一 WiFi，并使用电脑局域网 IP 访问'));
    };

    xhr.ontimeout = () => {
      if (stallTimer) clearInterval(stallTimer);
      reject(new Error('下载超时，请重试'));
    };

    onProgress?.(0, 0, false);
    resetStallTimer();
    xhr.send();
  });
}

async function fetchModelBuffer(modelUrl, onStatus) {
  const cached = await loadCachedModel();
  if (cached && cached.byteLength > 0) {
    onStatus?.(`使用本地缓存 (${(cached.byteLength / 1024 / 1024).toFixed(1)} MB)`);
    return cached;
  }

  onStatus?.('开始下载模型…');

  const buffer = await downloadArrayBuffer(modelUrl, (loaded, total, stalled) => {
    if (stalled && loaded > 0) {
      onStatus?.(`${formatProgress(loaded, total)}（仍在传输，请稍候）`);
      return;
    }
    onStatus?.(formatProgress(loaded, total));
  });

  await saveCachedModel(buffer);
  return buffer;
}

async function createSessionWithProgress(modelBuffer, onStatus) {
  let seconds = 0;
  const timer = setInterval(() => {
    seconds += 1;
    onStatus?.(`解析 ONNX 模型… ${seconds}s`);
  }, 1000);

  try {
    return await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
      executionProviders: ['wasm'],
    });
  } finally {
    clearInterval(timer);
  }
}

async function initDepthModel({ modelUrl, wasmBase, onStatus, runtimeConfig = null }) {
  if (typeof ort === 'undefined') {
    throw new Error('ONNX Runtime 未加载：请确认 vendor/onnxruntime-web/ort.wasm.min.js 可访问');
  }

  if (runtimeConfig) {
    inputSize = runtimeConfig.inputSize || inputSize;
    modelCacheKey = runtimeConfig.cacheKey || modelCacheKey;
    forceSingleThread = !!runtimeConfig.forceSingleThread;
  }

  infernoLUT = buildInfernoLUT();
  const numThreads = configureOrt(wasmBase);
  const modeLabel = forceSingleThread
    ? '单线程（1Hz 模式）'
    : numThreads > 1
      ? `${numThreads} 线程 WASM`
      : '单线程 WASM';
  onStatus?.(`加载 ${inputSize}×${inputSize} 模型，${modeLabel}`);
  await verifyWasmRuntime(wasmBase, onStatus);

  const modelBuffer = await fetchModelBuffer(modelUrl, onStatus);
  session = await createSessionWithProgress(modelBuffer, onStatus);
  onStatus?.('模型就绪');
  return { session, numThreads, inputSize };
}

function preprocessFromVideo(video) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) {
    throw new Error('视频尚未就绪');
  }

  const canvas = document.createElement('canvas');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  const tensor = new Float32Array(1 * 3 * inputSize * inputSize);
  const plane = inputSize * inputSize;
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
  small.width = inputSize;
  small.height = inputSize;
  const sctx = small.getContext('2d');
  const imageData = sctx.createImageData(inputSize, inputSize);
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
  const input = new ort.Tensor('float32', tensor, [1, 3, inputSize, inputSize]);

  const start = performance.now();
  const outputs = await session.run({ [inputName]: input });
  const elapsedMs = performance.now() - start;

  const depth = outputs[session.outputNames[0]].data;
  const imageData = depthToImageData(depth, width, height);
  return { imageData, elapsedMs, width, height };
}

window.DepthEngine = {
  loadRuntimeConfig,
  initDepthModel,
  inferDepthFromVideo,
};
