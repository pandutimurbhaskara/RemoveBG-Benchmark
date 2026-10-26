import type { GpuInfo, ModelKey } from './models'

export type Backend = 'webgpu' | 'wasm'

export type LoadStage = 'detecting' | 'not-downloaded' | 'downloading' | 'initializing' | 'ready' | 'error'

export interface FileProgress {
  file: string
  loaded: number
  total: number
  done: boolean
}

export interface Timings {
  preprocessMs: number
  inferenceMs: number
  postprocessMs: number
  totalMs: number
}

/**
 * What each model has in the browser cache: all its files, and whether the files this device
 * would load (GPU or CPU weights) are all there, so it can start without downloading.
 */
export type CacheInfo = Record<ModelKey, { files: number; bytes: number; ready: boolean }>

/** Messages sent from the UI thread to the worker. */
export type WorkerRequest =
  // Without `download`, a model whose files aren't cached is not fetched; the worker replies `needsDownload`.
  | { type: 'load'; model: ModelKey; download: boolean; forceCpu?: boolean; fallbackReason?: string }
  | { type: 'run'; id: number; image: Blob }
  // Skip a queued run (its image was removed).
  | { type: 'cancel'; id: number }
  | { type: 'deleteCache'; model: ModelKey }
  | { type: 'cacheInfo'; forceCpu: boolean }

/** Messages sent from the worker back to the UI thread. */
export type WorkerResponse =
  | { type: 'stage'; stage: LoadStage; backend?: Backend; dtype?: string; note?: string }
  | { type: 'progress'; files: FileProgress[]; loaded: number; total: number }
  | { type: 'ready'; model: ModelKey; backend: Backend; dtype: string; loadMs: number; fallbackReason?: string }
  // `mask` is the model's raw soft mask (one byte per pixel, 0 = background, 255 = foreground)
  // at the original image size; the UI applies cleanup and composites it.
  | {
      type: 'result'
      id: number
      model: ModelKey
      backend: Backend
      dtype: string
      mask: Uint8ClampedArray
      width: number
      height: number
      timings: Timings
      /** Wall-clock times (ms since epoch) when the worker started and finished this run. */
      startedAt: number
      finishedAt: number
    }
  | { type: 'cache'; info: CacheInfo }
  | { type: 'gpu'; info: GpuInfo }
  // WebGPU failed mid-load. ONNX Runtime can be left unusable in this worker afterwards,
  // so the UI starts a fresh worker and retries on CPU there.
  | { type: 'restartOnCpu'; model: ModelKey; reason: string; download: boolean }
  // The model's files for this device aren't cached and the load didn't allow downloading.
  | { type: 'needsDownload'; model: ModelKey; backend: Backend; dtype: string }
  | { type: 'error'; id?: number; message: string; finishedAt?: number }
