# On-device background removal

React + Vite + TypeScript app that removes image backgrounds entirely in the browser with
[Transformers.js](https://huggingface.co/docs/transformers.js).

Models (measured on an Apple Silicon Mac in Chrome, 800×800 product photo with text and props behind it):

| Model | License | WebGPU | CPU (WASM) | Notes |
|---|---|---|---|---|
| [BiRefNet lite 1024, WebGPU-patched](https://huggingface.co/jiabins0303/birefnet-lite-1024-webgpu) | MIT | ✅ ~4.5 s, 115 MB | ❌ | Cleanest result. Original BiRefNet weights; only the ONNX graph was rewritten |
| [MVANet](https://huggingface.co/onnx-community/MVANet-ONNX) | MIT | ✅ ~3.5 s, 85 MB (q4f16) | ✅ ~16 s, 141 MB | Clean; background shows through see-through parts (fan grill) |
| [InSPyReNet Swin-B](https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX) | MIT | ✅ ~4 s, 199 MB | ❌ (fp16 gives an empty mask) | Clean |
| [IS-Net general](https://huggingface.co/Ko033/isnet-general-use-onnx) | Apache-2.0 (code)* | ✅ ~0.7 s, 88 MB | ✅ ~3 s, 46 MB | RMBG-1.4's architecture and speed; can keep background props |
| [BiRefNet lite 512](https://huggingface.co/studioludens/birefnet-lite-512) (default) | MIT | ✅ ~1.6–2.5 s, 99 MB | ✅ ~2 s, 192 MB | Runs everywhere; less detail |
| [BEN2](https://huggingface.co/onnx-community/BEN2-ONNX) | MIT | ❌ on Apple GPUs | ✅ ~16 s, 219 MB | Slow |
| [MODNet](https://huggingface.co/Xenova/modnet) | Apache-2.0 | ✅ ~0.2 s, 13 MB | ✅ 7 MB | People/portraits only |
| [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | **Non-commercial** | ✅ ~0.7 s, 88 MB | ✅ ~2.6 s, 44 MB | Commercial use needs a BRIA license |

\* The DIS project licenses its code under Apache-2.0; the `isnet-general-use` weights have no explicit license and
were trained on the DIS5K dataset, which has its own terms of use. Check before commercial use.

Models can be switched, downloaded and deleted from the page.
Models are registered in `src/models.ts`.

- Models download only when you click **Download** (with a progress bar) and are cached by the browser. Picking a model
  that isn't downloaded shows a download prompt instead of fetching it; on the next visit the last-used model loads from
  the cache. Each model card has separate **Download** and **Delete** buttons.
- Runs on **WebGPU** (fp16 when the GPU supports `shader-f16`, else fp32), and falls back to **WASM/CPU** if WebGPU is unavailable or fails. A failed WebGPU attempt can leave ONNX Runtime unusable, so the fallback runs in a fresh worker.
- Inference runs in a Web Worker so the UI stays responsive.
- Upload several images at once (or drop/paste them). Each image keeps its result per model, so switching images or models
  shows earlier results instantly; images missing a result for the current model are queued. Remove an image to cancel
  its queued run.
- Download one result as PNG, all results for the current model as a ZIP, or export a JSON log of every model load and run
  (file, model, backend, precision, timings, timestamps, errors) for benchmarking.
- A "Background cleanup" slider removes the faint traces the model leaves on busy backgrounds (`src/composite.ts`). It re-applies without re-running the model.
- "Keep main subject only" (on by default) drops separate blobs smaller than 25% of the main subject, such as promo text, logos or props beside a product. With it, BiRefNet lite 512, MVANet and IS-Net matched RMBG-1.4 on our product-photo test set.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build && npm run preview
```

Add `?device=wasm` to the URL to force the CPU path for comparison.

## Deploying to Cloudflare (free tier)

The app is a static site, deployed as a static-assets Worker (`wrangler.jsonc`, serves `dist/`).

- **From GitHub (auto-deploy on push):** Cloudflare dashboard → Workers & Pages → Create → Import a repository →
  pick this repo. Build command `npm run build`, deploy command `npx wrangler deploy`.
- **From your machine:** `npx wrangler login` once, then `npm run deploy`.
- `npm run preview:cf` serves the production build locally exactly as Cloudflare will.

`public/_headers` sets the cross-origin isolation headers (multi-threaded CPU fallback) and long caching for hashed
assets. onnxruntime-web's 26.9 MB WASM is left out of the build (Cloudflare's per-file limit is 25 MiB); Transformers.js
loads it from the jsDelivr CDN at runtime, and models come from huggingface.co, so both hosts must be reachable.

**License note:** RMBG-1.4 weights are for non-commercial use; the other models allow commercial use. To add a model, add an entry to `src/models.ts` with its precisions and sizes.
