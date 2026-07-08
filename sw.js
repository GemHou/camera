const CACHE = 'cqmx-v1';
const URLS = [
  '.', 'index.html', 'crop_beauty.js', 'manifest.json',
  'vendor/onnxruntime-web/ort.wasm.min.js',
  'vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(URLS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => e.waitUntil(clients.claim()));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.includes('vendor/') || url.pathname.includes('.html') || url.pathname.includes('.js') || url.pathname.includes('.wasm') || url.pathname.includes('.json')) {
    e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
  }
});
