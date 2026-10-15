// Downloads that carry on after the person leaves the page or closes the tab (Background Fetch, Chromium
// browsers; see src/runtime/background-download.ts). The browser downloads the files itself, showing its own
// progress notification; when they arrive, this keeps them in a cache, where the app's model store finds them
// on the next load and moves them into its own storage (src/store/fetch-artifact.ts). Handles nothing else: no
// fetch handler, so the app's own requests never pass through here.

const CACHE = 'quick-inf-background-downloads';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

/** Keeps every file that arrived whole (also after a partial failure: those needn't be downloaded again). */
async function keep(registration) {
  const cache = await caches.open(CACHE);
  let kept = 0;
  for (const record of await registration.matchAll()) {
    try {
      const response = await record.responseReady;
      if (response.ok) {
        await cache.put(record.request, response);
        kept++;
      }
    } catch {
      // Not downloaded (or not whole): the app downloads it itself.
    }
  }
  return kept;
}

async function tell(message) {
  for (const client of await self.clients.matchAll({ includeUncontrolled: true })) client.postMessage(message);
}

self.addEventListener('backgroundfetchsuccess', (event) => {
  event.waitUntil(
    (async () => {
      await keep(event.registration);
      await event.updateUI({ title: 'Model downloaded. Open Quick Inference to start it.' });
      await tell({ kind: 'background-download', id: event.registration.id, state: 'done' });
    })(),
  );
});

self.addEventListener('backgroundfetchfail', (event) => {
  event.waitUntil(
    (async () => {
      await keep(event.registration);
      await tell({ kind: 'background-download', id: event.registration.id, state: 'failed', reason: event.registration.failureReason });
    })(),
  );
});

self.addEventListener('backgroundfetchabort', (event) => {
  event.waitUntil(tell({ kind: 'background-download', id: event.registration.id, state: 'aborted' }));
});

// Tapping the browser's download notification opens the model in the app.
self.addEventListener('backgroundfetchclick', (event) => {
  const modelId = event.registration.id.replace(/^model:/, '');
  event.waitUntil(self.clients.openWindow(`/${encodeURIComponent(modelId)}/`));
});
