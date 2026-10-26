/// <reference lib="webworker" />
import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
  type PreTrainedModel,
  type Processor,
  type ProgressInfo,
} from '@huggingface/transformers'
import type { Backend, CacheInfo, FileProgress, WorkerRequest, WorkerResponse } from './messages'
import { MODELS, gpuProblem, onnxFile, plannedRun, unsupportedReason, webgpuDtype, type Dtype, type GpuInfo, type ModelKey, type ModelSpec } from './models'

// Always fetch from the Hugging Face Hub and keep the files in the browser cache,
// so the download only happens on the first visit.
env.allowLocalModels = false
env.useBrowserCache = true

const post = (msg: WorkerResponse, transfer: Transferable[] = []) =>
  (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer)

interface Loaded {
  spec: ModelSpec
  backend: Backend
  dtype: string
  model: PreTrainedModel
  processor: Processor
}

let current: Loaded | null = null

// Runs the UI no longer wants (their image was removed); skipped when their turn comes.
const cancelled = new Set<number>()

// Loads, cache deletions and runs are executed one at a time, in the order they arrive.
let queue: Promise<unknown> = Promise.resolve()
const enqueue = (task: () => Promise<void>) => {
  queue = queue.then(task, task)
}

const files = new Map<string, FileProgress>()

function onProgress(info: ProgressInfo) {
  if (!('file' in info)) return
  const prev = files.get(info.file) ?? { file: info.file, loaded: 0, total: 0, done: false }

  if (info.status === 'progress') {
    files.set(info.file, { ...prev, loaded: info.loaded, total: info.total })
  } else if (info.status === 'done') {
    files.set(info.file, { ...prev, loaded: prev.total || prev.loaded, done: true })
  } else {
    files.set(info.file, prev)
  }

  const list = [...files.values()]
  post({
    type: 'progress',
    files: list,
    loaded: list.reduce((sum, f) => sum + f.loaded, 0),
    total: list.reduce((sum, f) => sum + f.total, 0),
  })
}

let gpuInfo: Promise<GpuInfo> | null = null

function probeGpu(): Promise<GpuInfo> {
  gpuInfo ??= (async (): Promise<GpuInfo> => {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
    if (!gpu) return { available: false, reason: 'WebGPU is not supported by this browser' }
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) return { available: false, reason: 'no WebGPU adapter is available' }
      return {
        available: true,
        fp16: adapter.features.has('shader-f16'),
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      }
    } catch (err) {
      return { available: false, reason: `the WebGPU adapter request failed: ${errorMessage(err)}` }
    }
  })()
  return gpuInfo
}

async function loadOn(spec: ModelSpec, backend: Backend, dtype: string): Promise<Loaded> {
  files.clear()
  post({ type: 'stage', stage: 'downloading', backend, dtype: spec.precisionLabel ?? dtype })

  const [model, processor] = await Promise.all([
    AutoModel.from_pretrained(spec.id, {
      config: spec.config as never,
      device: backend,
      dtype: dtype as never,
      model_file_name: spec.modelFileName,
      progress_callback: onProgress,
    }),
    AutoProcessor.from_pretrained(spec.id, { progress_callback: onProgress }),
  ])

  post({ type: 'stage', stage: 'initializing', backend, dtype: spec.precisionLabel ?? dtype, note: 'Warming up the model' })
  try {
    // Run once on a tiny image so shader compilation / memory allocation
    // doesn't get counted in the first real measurement.
    const blank = new RawImage(new Uint8ClampedArray(32 * 32 * 3), 32, 32, 3)
    const { pixel_values } = await processor(blank)
    await model({ [model.sessions.model.inputNames[0]]: pixel_values })
  } catch (err) {
    await model.dispose()
    throw err
  }

  return { spec, backend, dtype: spec.precisionLabel ?? dtype, model, processor }
}

async function load(key: ModelKey, download: boolean, forceCpu: boolean, cpuReason?: string) {
  const spec = MODELS[key]
  const start = performance.now()

  post({ type: 'stage', stage: 'detecting' })
  const gpu = await probeGpu()
  const problem = forceCpu ? (cpuReason ?? 'CPU forced via ?device=wasm') : gpuProblem(spec, gpu)

  // Never download implicitly: if this device's files aren't cached, ask the user first.
  const plan = plannedRun(spec, gpu, forceCpu)
  if (!plan) throw new Error(unsupportedReason(spec, gpu, forceCpu) ?? `${spec.label} needs WebGPU.`)
  if (!download && !(await isCached(spec, plan.dtype))) {
    post({ type: 'needsDownload', model: key, backend: plan.backend, dtype: spec.precisionLabel ?? plan.dtype })
    return
  }

  // Free the previous model (and its GPU memory) before loading the next one.
  if (current) {
    await current.model.dispose()
    current = null
  }

  if (!problem && gpu.available) {
    const dtype = webgpuDtype(spec, gpu.fp16)!
    try {
      current = await loadOn(spec, 'webgpu', dtype)
      post({ type: 'ready', model: key, backend: 'webgpu', dtype: spec.precisionLabel ?? dtype, loadMs: performance.now() - start })
      return
    } catch (err) {
      // A failed download is not a GPU problem: report it instead of retrying on CPU.
      if (isDownloadError(err)) throw new Error(downloadFailedMessage(err))
      console.warn('[bg-removal] WebGPU failed', err)
      if (!spec.dtype.wasm) throw new Error(`${spec.label} failed on WebGPU and has no CPU version: ${errorMessage(err)}`)
      post({ type: 'restartOnCpu', model: key, reason: `WebGPU failed: ${errorMessage(err)}`, download })
      return
    }
  }

  const dtype = spec.dtype.wasm
  if (!dtype) throw new Error(unsupportedReason(spec, gpu, forceCpu) ?? `${spec.label} needs WebGPU.`)
  try {
    current = await loadOn(spec, 'wasm', dtype)
  } catch (err) {
    throw isDownloadError(err) ? new Error(downloadFailedMessage(err)) : err
  }
  post({ type: 'ready', model: key, backend: 'wasm', dtype, loadMs: performance.now() - start, fallbackReason: problem ?? undefined })
}

async function removeBackground(id: number, blob: Blob) {
  if (!current) throw new Error('Model is not loaded yet')
  const { spec, backend, dtype, model, processor } = current

  const startedAt = Date.now()
  const t0 = performance.now()
  const image = await RawImage.fromBlob(blob)
  const { pixel_values } = await processor(image)
  const t1 = performance.now()

  // Models name their single input/output differently, so use whatever the ONNX session declares.
  const { inputNames, outputNames } = model.sessions.model
  const outputs = await model({ [inputNames[0]]: pixel_values })
  const t2 = performance.now()

  // Some models output probabilities, others raw logits; apply a sigmoid only to the latter.
  let prediction = outputs[outputNames[0]][0]
  if (prediction.data.some((x: number) => x < -1e-5 || x > 1 + 1e-5)) prediction = prediction.sigmoid()
  const mask = await RawImage.fromTensor(prediction.mul(255).to('uint8')).resize(image.width, image.height)
  const maskData = new Uint8ClampedArray(mask.data)
  const t3 = performance.now()

  post(
    {
      type: 'result',
      id,
      model: spec.key,
      backend,
      dtype,
      mask: maskData,
      width: image.width,
      height: image.height,
      timings: {
        preprocessMs: t1 - t0,
        inferenceMs: t2 - t1,
        postprocessMs: t3 - t2,
        totalMs: t3 - t0,
      },
      startedAt,
      finishedAt: Date.now(),
    },
    [maskData.buffer],
  )
}

// transformers.js stores each downloaded file in Cache Storage, keyed by its Hub URL.
const cachePrefix = (spec: ModelSpec) => `${env.remoteHost}${spec.id}/`

/** Whether the files needed to run `spec` at `dtype` are all in the browser cache. */
async function isCached(spec: ModelSpec, dtype: Dtype) {
  if (typeof caches === 'undefined') return false
  const cache = await caches.open(env.cacheKey)
  const base = `${cachePrefix(spec)}resolve/main/`
  const required = [onnxFile(spec, dtype), 'preprocessor_config.json']
  for (const file of required) {
    if (!(await cache.match(base + file))) return false
  }
  return true
}

// The page's CPU-only setting, used to decide which files count as "ready" for this device.
let pageForceCpu = false

async function readCacheInfo(): Promise<CacheInfo> {
  const info = Object.fromEntries(
    Object.keys(MODELS).map((key) => [key, { files: 0, bytes: 0, ready: false }]),
  ) as CacheInfo
  if (typeof caches === 'undefined') return info

  const gpu = await probeGpu()
  for (const spec of Object.values(MODELS)) {
    const plan = plannedRun(spec, gpu, pageForceCpu)
    info[spec.key].ready = plan ? await isCached(spec, plan.dtype) : false
  }

  const cache = await caches.open(env.cacheKey)
  for (const request of await cache.keys()) {
    const spec = Object.values(MODELS).find((s) => request.url.startsWith(cachePrefix(s)))
    if (!spec) continue
    const response = await cache.match(request)
    info[spec.key].files += 1
    info[spec.key].bytes += Number(response?.headers.get('content-length') ?? 0)
  }
  return info
}

async function deleteCache(key: ModelKey) {
  if (typeof caches === 'undefined') return
  const prefix = cachePrefix(MODELS[key])
  const cache = await caches.open(env.cacheKey)
  for (const request of await cache.keys()) {
    if (request.url.startsWith(prefix)) await cache.delete(request)
  }
}

async function postCacheInfo() {
  try {
    post({ type: 'cache', info: await readCacheInfo() })
  } catch (err) {
    console.warn('[bg-removal] Could not read the model cache', err)
  }
}

function errorMessage(err: unknown) {
  return err instanceof Error ? err.message : String(err)
}

/**
 * Whether a load failed while fetching model files (network, CORS or HTTP error) rather than in
 * the runtime. transformers.js reports HTTP errors with the file's URL in the message.
 */
function isDownloadError(err: unknown) {
  if (!(err instanceof Error)) return false
  return (
    err.name === 'ModelFileNotFoundError' ||
    (err instanceof TypeError && /fetch|network|load failed/i.test(err.message)) ||
    /"https?:\/\//.test(err.message)
  )
}

function downloadFailedMessage(err: unknown) {
  return `Couldn't download the model files from huggingface.co (${errorMessage(err)}). Check your connection and try again.`
}

probeGpu().then((info) => post({ type: 'gpu', info }))

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data
  switch (msg.type) {
    case 'load':
      enqueue(async () => {
        try {
          await load(msg.model, msg.download, msg.forceCpu ?? false, msg.fallbackReason)
        } catch (err) {
          post({ type: 'stage', stage: 'error' })
          post({ type: 'error', message: errorMessage(err) })
        }
        await postCacheInfo()
      })
      break

    case 'run':
      enqueue(async () => {
        if (cancelled.delete(msg.id)) return
        try {
          await removeBackground(msg.id, msg.image)
        } catch (err) {
          post({ type: 'error', id: msg.id, message: errorMessage(err), finishedAt: Date.now() })
        }
      })
      break

    case 'cancel':
      cancelled.add(msg.id)
      break

    case 'deleteCache':
      enqueue(async () => {
        try {
          await deleteCache(msg.model)
        } catch (err) {
          post({ type: 'error', message: `Could not delete the cache: ${errorMessage(err)}` })
        }
        await postCacheInfo()
      })
      break

    case 'cacheInfo':
      pageForceCpu = msg.forceCpu
      enqueue(postCacheInfo)
      break
  }
})
