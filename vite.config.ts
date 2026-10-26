import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

// Cross-origin isolation enables SharedArrayBuffer, which lets the WASM (CPU)
// fallback run multi-threaded. `credentialless` still allows loading the model
// from the Hugging Face Hub. Your production host needs the same headers.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

// Transformers.js always loads onnxruntime-web's WASM from the jsDelivr CDN at runtime, but Vite
// still emits the 26.9 MB copy the library references. It is never fetched, and it is over
// Cloudflare's 25 MiB per-file limit, so leave it out of the build.
const dropUnusedOrtWasm = (): Plugin => ({
  name: 'drop-unused-ort-wasm',
  apply: 'build',
  generateBundle(_, bundle) {
    for (const fileName of Object.keys(bundle)) {
      if (/ort-wasm[^/]*\.wasm$/.test(fileName)) delete bundle[fileName]
    }
  },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), dropUnusedOrtWasm()],
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  worker: {
    format: 'es',
    // The model runs in a worker, so its bundle is where the WASM asset is emitted.
    plugins: () => [dropUnusedOrtWasm()],
  },
  optimizeDeps: {
    // Transformers.js loads onnxruntime-web's WASM files at runtime; pre-bundling breaks those paths.
    exclude: ['@huggingface/transformers'],
  },
})
