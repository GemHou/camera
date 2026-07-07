/* CQ + MX — browser-side ONNX inference with XHR + IndexedDB cache */

const CB_MEAN = [0.485, 0.456, 0.406];
const CB_STD = [0.229, 0.224, 0.225];
const CROP_INPUT = { w: 480, h: 270 };
const BEAUTY_INPUT = { w: 480, h: 270 };

const DB_NAME = 'camera-cb-cache';
const DB_STORE = 'models';
const CACHE_KEY = 'cb_models_v2';

let cropSession = null;
let beautySession = null;

function configureCbOrt(wasmBase) {
  ort.env.wasm.wasmPaths = wasmBase;
  ort.env.wasm.simd = true;
  const canMT = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  ort.env.wasm.numThreads = canMT ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
}

function openDb() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB_NAME, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(DB_STORE);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function loadCached(key) {
  try {
    const db = await openDb();
    return await new Promise((r) => {
      const tx = db.transaction(DB_STORE, 'readonly');
      const g = tx.objectStore(DB_STORE).get(key);
      g.onsuccess = () => r(g.result || null);
      g.onerror = () => r(null);
    });
  } catch { return null; }
}

async function saveCache(key, buf) {
  try {
    const db = await openDb();
    await new Promise((r) => {
      const tx = db.transaction(DB_STORE, 'readwrite');
      tx.objectStore(DB_STORE).put(buf, key);
      tx.oncomplete = r;
    });
  } catch (e) { console.warn('HCSB:', e); }
}

function downloadXHR(url, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.timeout = 10 * 60 * 1000;
    let last = 0, stallTimer;

    const resetStall = () => {
      clearInterval(stallTimer);
      stallTimer = setInterval(() => onProgress?.(last, xhr.total || 0, true), 3000);
    };

    xhr.onprogress = (e) => {
      last = e.loaded;
      onProgress?.(e.loaded, e.lengthComputable ? e.total : xhr.total || 0, false);
      resetStall();
    };
    xhr.onload = () => { clearInterval(stallTimer); if (xhr.status >= 200 && xhr.status < 300 && xhr.response) resolve(xhr.response); else reject(new Error(`HTTP ${xhr.status}`)); };
    xhr.onerror = () => { clearInterval(stallTimer); reject(new Error('WLCW')); };
    xhr.ontimeout = () => { clearInterval(stallTimer); reject(new Error('XZCS')); };
    onProgress?.(0, 0, false);
    resetStall();
    xhr.send();
  });
}

function formatProgress(loaded, total) {
  const m = (loaded / 1024 / 1024).toFixed(1);
  if (total > 0) return `${Math.round(loaded / total * 100)}% (${m} MB)`;
  return `${m} MB…`;
}

async function loadModelWithCache(url, cacheKey, onStatus) {
  // 1. try cache
  const cached = await loadCached(cacheKey);
  if (cached) { onStatus?.('BDHC'); return cached; }

  // 2. download with XHR
  onStatus?.('XZMX…');
  const buf = await downloadXHR(url, (loaded, total, stalled) => {
    if (stalled && loaded > 0) { onStatus?.(`${formatProgress(loaded, total)}（RZCCS）`); return; }
    onStatus?.(formatProgress(loaded, total));
  });

  // 3. cache
  await saveCache(cacheKey, buf);
  return buf;
}

async function createSessionFromBuffer(buf, onStatus) {
  let s = 0;
  const t = setInterval(() => { s++; onStatus?.(`JXMX… ${s}s`); }, 1000);
  try {
    return await ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ['wasm'] });
  } finally { clearInterval(t); }
}

async function initCropModel({ modelUrl, onStatus }) {
  onStatus?.('JZCQMX…');
  const key = CACHE_KEY + '_crop';
  try {
    const buf = await loadModelWithCache(modelUrl, key, onStatus);
    cropSession = await createSessionFromBuffer(buf, onStatus);
  } catch (e) {
    console.error('Crop MX JZSB:', e);
    throw e;
  }
  onStatus?.('CQMXJX');
}

async function initCropFromBuffer(buf, { onStatus }) {
  cropSession = await createSessionFromBuffer(buf, onStatus);
  await saveCache(CACHE_KEY + '_crop', buf);
  onStatus?.('CQMXJX');
}

async function initBeautyFromBuffer(buf, { onStatus }) {
  beautySession = await createSessionFromBuffer(buf, onStatus);
  await saveCache(CACHE_KEY + '_beauty', buf);
  onStatus?.('MXMXJX');
}

async function initCropFromFile(file, { onStatus }) {
  const buf = await file.arrayBuffer();
  await initCropFromBuffer(buf, { onStatus });
}

async function initBeautyFromFile(file, { onStatus }) {
  const buf = await file.arrayBuffer();
  await initBeautyFromBuffer(buf, { onStatus });
}

async function initBeautyModel({ modelUrl, onStatus }) {
  onStatus?.('JZMXMX…');
  const key = CACHE_KEY + '_beauty';
  try {
    const buf = await loadModelWithCache(modelUrl, key, onStatus);
    beautySession = await createSessionFromBuffer(buf, onStatus);
  } catch (e) {
    console.error('Beauty MX JZSB:', e);
    throw e;
  }
  onStatus?.('MXMXJX');
}

function cbPreprocessFrame(video, targetW, targetH) {
  const c = document.createElement('canvas');
  c.width = targetW; c.height = targetH;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, targetW, targetH);
  const { data } = ctx.getImageData(0, 0, targetW, targetH);
  const plane = targetW * targetH;
  const t = new Float32Array(1 * 3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i*4]/255, g = data[i*4+1]/255, b = data[i*4+2]/255;
    t[i] = (r - CB_MEAN[0]) / CB_STD[0];
    t[plane+i] = (g - CB_MEAN[1]) / CB_STD[1];
    t[2*plane+i] = (b - CB_MEAN[2]) / CB_STD[2];
  }
  return t;
}

function cbPreprocessCrop(video, bbox, targetW, targetH) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const bx = [Math.round(bbox[0]*vw), Math.round(bbox[1]*vh), Math.round(bbox[2]*vw), Math.round(bbox[3]*vh)];
  const off = document.createElement('canvas');
  off.width = vw; off.height = vh;
  off.getContext('2d').drawImage(video, 0, 0);
  const c = document.createElement('canvas');
  c.width = targetW; c.height = targetH;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(off, bx[0], bx[1], bx[2]-bx[0], bx[3]-bx[1], 0, 0, targetW, targetH);
  const { data } = ctx.getImageData(0, 0, targetW, targetH);
  const plane = targetW * targetH;
  const t = new Float32Array(1 * 3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i*4]/255, g = data[i*4+1]/255, b = data[i*4+2]/255;
    t[i] = (r - CB_MEAN[0]) / CB_STD[0];
    t[plane+i] = (g - CB_MEAN[1]) / CB_STD[1];
    t[2*plane+i] = (b - CB_MEAN[2]) / CB_STD[2];
  }
  return t;
}

async function runCropAndBeauty(video) {
  if (!cropSession || !beautySession) throw new Error('MXWJJX');

  const ct = cbPreprocessFrame(video, CROP_INPUT.w, CROP_INPUT.h);
  const ci = new ort.Tensor('float32', ct, [1, 3, CROP_INPUT.h, CROP_INPUT.w]);
  const t0 = performance.now();
  const co = await cropSession.run({ pixel_values: ci });
  const cropMs = performance.now() - t0;
  const bbox = co.pred_bbox.data;

  const bt = cbPreprocessCrop(video, bbox, BEAUTY_INPUT.w, BEAUTY_INPUT.h);
  const bi = new ort.Tensor('float32', bt, [1, 3, BEAUTY_INPUT.h, BEAUTY_INPUT.w]);
  const t1 = performance.now();
  const bo = await beautySession.run({ input: bi });
  const beautyMs = performance.now() - t1;
  const s = bo.scores.data;

  return {
    bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
    cropMs, beautyMs, totalMs: cropMs + beautyMs,
    total: s[0]*100, element: s[1]*100, story: s[2]*100,
    composition: s[3]*100, light: s[4]*100, atmosphere: s[5]*100,
  };
}

window.CropBeautyEngine = {
  initCropModel, initBeautyModel,
  initCropFromFile, initBeautyFromFile,
  initCropFromBuffer, initBeautyFromBuffer,
  loadCached, configureCbOrt,
  runCropAndBeauty,
};
