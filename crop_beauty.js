/* Crop + Beauty Score — browser-side ONNX inference */

const CB_MEAN = [0.485, 0.456, 0.406];
const CB_STD = [0.229, 0.224, 0.225];
const CROP_INPUT = { w: 480, h: 270 };
const BEAUTY_INPUT = { w: 480, h: 270 };

let cropSession = null;
let beautySession = null;

function configureCbOrt(wasmBase) {
  ort.env.wasm.wasmPaths = wasmBase;
  ort.env.wasm.simd = true;
  const canMT = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  const cores = navigator.hardwareConcurrency || 4;
  ort.env.wasm.numThreads = canMT ? Math.min(4, cores) : 1;
}

async function loadCbModel(url, onStatus) {
  let seconds = 0;
  const timer = setInterval(() => { seconds++; onStatus?.(`JXMX… ${seconds}s`); }, 1000);
  try {
    return await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  } finally {
    clearInterval(timer);
  }
}

async function initCropModel({ wasmBase, modelUrl, modelBase, onStatus }) {
  configureCbOrt(wasmBase);
  // try: INT8 from Pages → FP32 from R2 (or local)
  const pagesUrl = modelUrl;  // relative to site root (CF Pages or local)
  const r2Url = modelBase ? modelBase + '/' + modelUrl.split('/').pop() : modelUrl;
  const i8Url = pagesUrl.replace('.onnx', '_int8.onnx');
  onStatus?.('JZCQMX…');
  try {
    cropSession = await loadCbModel(i8Url, onStatus);
  } catch (e) {
    console.warn('INT8 SB, try R2/FP32:', e);
    cropSession = await loadCbModel(r2Url, onStatus);
  }
  onStatus?.('CQMXJX');
}

async function initBeautyModel({ wasmBase, modelUrl, modelBase, onStatus }) {
  const pagesUrl = modelUrl;
  const r2Url = modelBase ? modelBase + '/' + modelUrl.split('/').pop() : modelUrl;
  const i8Url = pagesUrl.replace('.onnx', '_int8.onnx');
  onStatus?.('JZMXMX…');
  try {
    beautySession = await loadCbModel(i8Url, onStatus);
  } catch (e) {
    console.warn('INT8 SB, try R2/FP32:', e);
    beautySession = await loadCbModel(r2Url, onStatus);
  }
  onStatus?.('MXMXJX');
}

function cbPreprocessFrame(video, targetW, targetH) {
  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, targetW, targetH);
  const { data } = ctx.getImageData(0, 0, targetW, targetH);

  const plane = targetW * targetH;
  const tensor = new Float32Array(1 * 3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    tensor[i] = (r - CB_MEAN[0]) / CB_STD[0];
    tensor[plane + i] = (g - CB_MEAN[1]) / CB_STD[1];
    tensor[2 * plane + i] = (b - CB_MEAN[2]) / CB_STD[2];
  }
  return tensor;
}

function cbPreprocessCrop(video, bbox, targetW, targetH) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const bx = [Math.round(bbox[0] * vw), Math.round(bbox[1] * vh),
              Math.round(bbox[2] * vw), Math.round(bbox[3] * vh)];

  const offscreen = document.createElement('canvas');
  offscreen.width = vw;
  offscreen.height = vh;
  const octx = offscreen.getContext('2d');
  octx.drawImage(video, 0, 0);

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = targetW;
  cropCanvas.height = targetH;
  const ctx = cropCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(offscreen, bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1],
                0, 0, targetW, targetH);
  const { data } = ctx.getImageData(0, 0, targetW, targetH);

  const plane = targetW * targetH;
  const tensor = new Float32Array(1 * 3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    tensor[i] = (r - CB_MEAN[0]) / CB_STD[0];
    tensor[plane + i] = (g - CB_MEAN[1]) / CB_STD[1];
    tensor[2 * plane + i] = (b - CB_MEAN[2]) / CB_STD[2];
  }
  return tensor;
}

async function runCropAndBeauty(video) {
  if (!cropSession || !beautySession) throw new Error('MXWJJX');

  // Step 1: Crop inference
  const cropTensor = cbPreprocessFrame(video, CROP_INPUT.w, CROP_INPUT.h);
  const cropInput = new ort.Tensor('float32', cropTensor, [1, 3, CROP_INPUT.h, CROP_INPUT.w]);
  const cropStart = performance.now();
  const cropOut = await cropSession.run({ pixel_values: cropInput });
  const cropMs = performance.now() - cropStart;
  const bbox = cropOut.pred_bbox.data;  // [x1, y1, x2, y2] ∈ [0,1]

  // Step 2: Beauty inference on cropped region
  const beautyTensor = cbPreprocessCrop(video, bbox, BEAUTY_INPUT.w, BEAUTY_INPUT.h);
  const beautyInput = new ort.Tensor('float32', beautyTensor, [1, 3, BEAUTY_INPUT.h, BEAUTY_INPUT.w]);
  const beautyStart = performance.now();
  const beautyOut = await beautySession.run({ input: beautyInput });
  const beautyMs = performance.now() - beautyStart;
  const scores = beautyOut.scores.data;

  return {
    bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
    cropMs, beautyMs, totalMs: cropMs + beautyMs,
    total: scores[0] * 100,
    element: scores[1] * 100,
    story: scores[2] * 100,
    composition: scores[3] * 100,
    light: scores[4] * 100,
    atmosphere: scores[5] * 100,
  };
}

window.CropBeautyEngine = { initCropModel, initBeautyModel, runCropAndBeauty };
