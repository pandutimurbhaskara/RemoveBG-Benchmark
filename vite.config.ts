import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Cross-origin isolation enables SharedArrayBuffer, which lets the WASM (CPU)
// fallback run multi-threaded. `credentialless` still allows loading the model
// from the Hugging Face Hub. Your production host needs the same headers.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // Transformers.js loads onnxruntime-web's WASM files at runtime; pre-bundling breaks those paths.
    exclude: ['@huggingface/transformers'],
  },
})
