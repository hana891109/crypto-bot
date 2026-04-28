// Service Worker - 背景持續掃描
const CACHE = 'yuan-v1';
let scanTimer = null;
let scanInterval = 5; // 分鐘

self.addEventListener('install', e => { self.skipWaiting(); });
self.addEventListener('activate', e => { e.waitUntil(clients.claim()); });

self.addEventListener('message', e => {
  const { type, data } = e.data || {};
  if (type === 'START_AUTO_SCAN') {
    scanInterval = data?.interval || 5;
    if (scanTimer) clearInterval(scanTimer);
    scanTimer = setInterval(() => {
      // 通知所有視窗執行掃描
      self.clients.matchAll().then(cls => {
        if (cls.length > 0) {
          cls[0].postMessage({ type: 'DO_SCAN' });
        }
      });
    }, scanInterval * 60 * 1000);
    console.log('[SW] 自動掃描啟動，間隔', scanInterval, '分鐘');
  }
  if (type === 'STOP_AUTO_SCAN') {
    if (scanTimer) clearInterval(scanTimer);
    scanTimer = null;
  }
  if (type === 'PING') {
    e.source.postMessage({ type: 'PONG', active: !!scanTimer });
  }
});
