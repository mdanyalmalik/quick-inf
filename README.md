# Quick Inference

Private AI chat and image generation that runs entirely in the browser: models download once to the device and run
on its GPU with WebGPU (or the CPU), with no server or account. Live at https://quickinf.com.

## Development

```sh
npm install        # also installs our ONNX Runtime Web build (see below)
npm run dev        # https://localhost:5173 (HTTPS and cross-origin isolation, as in production)
npm test           # unit and reference tests (vitest)
npm run build      # type-check and production build into dist/
```

Development pages, served by the dev server only:

- `diagnostics.html?model=<id>` checks a model against reference outputs from ONNX Runtime's CPU backend
  (`tools/reference/make_fixtures.py`).
- `bench.html?model=<id>` measures decoding and prompt speed; `tools/bench/bench.mjs` drives it in Chrome.
- `?sim=…` on the app simulates other devices and features (`src/device/simulation.ts`), e.g. `?sim=stream` for
  expert streaming.

The dev server serves model files from `tools/reference/.cache` when they're there (`tools/reference/fetch_model.py`,
`tools/bench/cache_model.py`), and from Hugging Face otherwise.

## Layout

- `src/catalogue/` the models: pinned files with sizes and SHA-256, prompt formats, quality notes.
- `src/runtime/` loading and running models in a worker: `onnx/` (ONNX Runtime adapter, load-time graph rewrites in
  `external-weights.ts`, expert streaming), `image/` (diffusion pipelines), sampling and context handling.
- `src/recommend/`, `src/device/` memory budgets and which models suit this device.
- `src/tokenizer/`, `src/conversations/`, `src/store/`, `src/ui/` tokenizers and chat templates, storage, the React UI.
- `tools/export/` exporters for our own ONNX conversions (published under huggingface.co/danyalmalik).

## ONNX Runtime

The app runs ONNX Runtime Web 1.30 with a few patches of ours (faster kernels), built from
https://github.com/mdanyalmalik/onnxruntime-web-kernels. `npm install` downloads the release pinned in
`tools/ort/release.json` over the npm package's WebGPU binaries, checking each file's SHA-256
(`QI_STOCK_ORT=1` keeps npm's). Its licence and notices are served at `/third-party/onnxruntime-web/`.

## Licence

Quick Inference is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE): you may use, change
and share it for any noncommercial purpose (personal use, research, hobby projects, and use by charities,
educational, research and public institutions). Commercial use needs a separate licence: ask the author.

Third-party code keeps its own licence: the npm dependencies theirs, ONNX Runtime Web MIT (notices in
`public/third-party/onnxruntime-web/`). The models are not part of this repository; each is downloaded from its own
repository under its own licence, which the app shows.
