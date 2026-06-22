/* Depth Anything V2 Small — ONNX inference in Web Worker */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js');

const INPUT_SIZE = 518;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;
let infernoLUT = null;

function buildInfernoLUT() {
  const anchors = [
    [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
    [186, 54, 85], [227, 89, 51], [249, 140, 10], [252, 255, 164]
  ];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (anchors.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;
    const a = anchors[Math.min(idx, anchors.length - 1)];
    const b = anchors[Math.min(idx + 1, anchors.length - 1)];
    lut[i * 3] = a[0] + (b[0] - a[0]) * frac;
    lut[i * 3 + 1] = a[1] + (b[1] - a[1]) * frac;
    lut[i * 3 + 2] = a[2] + (b[2] - a[2]) * frac;
  }
  return lut;
}

function preprocess(bitmap) {
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
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
  return tensor;
}

function postprocess(depthData, outWidth, outHeight) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < depthData.length; i++) {
    if (depthData[i] < min) min = depthData[i];
    if (depthData[i] > max) max = depthData[i];
  }
  const range = max - min + 1e-8;

  const small = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
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

  const large = new OffscreenCanvas(outWidth, outHeight);
  const lctx = large.getContext('2d');
  lctx.drawImage(small, 0, 0, outWidth, outHeight);
  return lctx.getImageData(0, 0, outWidth, outHeight);
}

async function runInference(bitmap, outWidth, outHeight) {
  const tensorData = preprocess(bitmap);
  bitmap.close();

  const inputName = session.inputNames[0];
  const input = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  const start = performance.now();
  const outputs = await session.run({ [inputName]: input });
  const elapsedMs = performance.now() - start;

  const depth = outputs[session.outputNames[0]].data;
  const result = postprocess(depth, outWidth, outHeight);

  return { imageData: result, elapsedMs };
}

self.onmessage = async (event) => {
  const msg = event.data;

  try {
    if (msg.type === 'init') {
      infernoLUT = buildInfernoLUT();
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
      ort.env.wasm.numThreads = Math.min(4, self.navigator?.hardwareConcurrency || 2);

      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: ['wasm'],
      });

      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'infer') {
      const { bitmap, width, height } = msg;
      const { imageData, elapsedMs } = await runInference(bitmap, width, height);
      self.postMessage(
        { type: 'result', elapsedMs, width, height, buffer: imageData.data.buffer },
        [imageData.data.buffer]
      );
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) });
  }
};
