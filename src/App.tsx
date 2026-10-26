import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { alphaCurve, applyMask, decodePixels, keepMainSubject, renderCutout } from './composite'
import { buildLog, cutoutFileName, saveFile, timestamp, zipFiles, type ModelLoadLog, type RunLog } from './export'
import type { Backend, CacheInfo, FileProgress, LoadStage, Timings, WorkerRequest, WorkerResponse } from './messages'
import { DEFAULT_MODEL, MODELS, isModelKey, plannedSizeMb, unsupportedReason, type GpuInfo, type ModelKey } from './models'

interface ModelState {
  key: ModelKey
  stage: LoadStage
  backend?: Backend
  dtype?: string
  note?: string
  files: FileProgress[]
  loaded: number
  total: number
  loadMs?: number
  fallbackReason?: string
  error?: string
}

/** One model's run on one image. */
interface RunResult {
  runId: number
  model: ModelKey
  status: 'queued' | 'done' | 'error'
  backend?: Backend
  mask?: Uint8ClampedArray
  width?: number
  height?: number
  timings?: Timings
  error?: string
}

/** An uploaded image and its results, kept per model so switching back shows them instantly. */
interface Item {
  id: number
  name: string
  blob: Blob
  originalUrl: string
  results: Partial<Record<ModelKey, RunResult>>
}

const MODEL_STORAGE_KEY = 'bg-removal:model'

function savedModel(): ModelKey {
  try {
    const value = localStorage.getItem(MODEL_STORAGE_KEY)
    return isModelKey(value) ? value : DEFAULT_MODEL
  } catch {
    return DEFAULT_MODEL
  }
}

const initialModelState = (key: ModelKey): ModelState => ({ key, stage: 'detecting', files: [], loaded: 0, total: 0 })

// `?device=wasm` skips WebGPU, handy for comparing against the CPU fallback.
const forceCpu = new URLSearchParams(location.search).get('device') === 'wasm'

const SAMPLE_URL =
  'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman_small.jpg'

const iso = (ms: number) => new Date(ms).toISOString()

export default function App() {
  const workerRef = useRef<Worker | null>(null)
  const nextItemId = useRef(1)
  const nextRunId = useRef(1)
  /** Which image and model each run belongs to, so worker replies can be routed back. */
  const runTargets = useRef(new Map<number, { itemId: number; model: ModelKey }>())
  const [initialModel] = useState(savedModel)
  /** The model most recently asked to load, for logging load errors. */
  const loadingModel = useRef<ModelKey>(initialModel)
  /** The model that is loaded and accepting runs, or null while one is loading. */
  const readyModel = useRef<ModelKey | null>(null)
  /** Latest image list, for queuing from worker events. */
  const itemsRef = useRef<Item[]>([])
  /** Models whose WebGPU load failed this session, with the reason; they load straight on CPU next time. */
  const gpuFailed = useRef(new Map<ModelKey, string>())
  /** `${itemId}:${model}` pairs already sent to the worker, so nothing is queued twice. */
  const queued = useRef(new Set<string>())
  const [model, setModel] = useState<ModelState>(() => initialModelState(initialModel))
  const [cache, setCache] = useState<CacheInfo | null>(null)
  const [gpu, setGpu] = useState<GpuInfo | null>(null)
  const [items, setItems] = useState<Item[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [cleanup, setCleanup] = useState(25)
  const [mainOnly, setMainOnly] = useState(true)
  const [modelLoads, setModelLoads] = useState<ModelLoadLog[]>([])
  const [runs, setRuns] = useState<RunLog[]>([])
  const [zipProgress, setZipProgress] = useState<string | null>(null)

  useEffect(() => {
    itemsRef.current = items
  }, [items])

  /** Sends images that have no result for `key` yet to the worker and records them as queued. */
  const enqueue = useCallback((candidates: Item[], key: ModelKey) => {
    const queuedAt = iso(Date.now())
    const assigned = new Map<number, number>()
    const newRuns: RunLog[] = []
    for (const item of candidates) {
      const pair = `${item.id}:${key}`
      if (item.results[key] || queued.current.has(pair)) continue
      queued.current.add(pair)
      const runId = nextRunId.current++
      assigned.set(item.id, runId)
      runTargets.current.set(runId, { itemId: item.id, model: key })
      workerRef.current?.postMessage({ type: 'run', id: runId, image: item.blob } satisfies WorkerRequest)
      newRuns.push({
        runId,
        imageId: item.id,
        file: { name: item.name, type: item.blob.type, bytes: item.blob.size },
        model: key,
        modelId: MODELS[key].id,
        status: 'queued',
        imageRemoved: false,
        queuedAt,
      })
    }
    if (newRuns.length === 0) return
    setItems((all) =>
      all.map((item) => {
        const runId = assigned.get(item.id)
        return runId ? { ...item, results: { ...item.results, [key]: { runId, model: key, status: 'queued' } } } : item
      }),
    )
    setRuns((all) => [...all, ...newRuns])
  }, [])

  useEffect(() => {
    const startWorker = () => {
      workerRef.current?.terminate()
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      worker.onmessage = onMessage
      workerRef.current = worker
      worker.postMessage({ type: 'cacheInfo', forceCpu } satisfies WorkerRequest)
      return worker
    }

    const updateRun = (runId: number, patch: Partial<RunLog>) =>
      setRuns((all) => all.map((r) => (r.runId === runId ? { ...r, ...patch } : r)))

    const updateResult = (runId: number, patch: Partial<RunResult>) => {
      const target = runTargets.current.get(runId)
      if (!target) return
      setItems((all) =>
        all.map((item) => {
          const result = item.results[target.model]
          if (item.id !== target.itemId || result?.runId !== runId) return item
          return { ...item, results: { ...item.results, [target.model]: { ...result, ...patch } } }
        }),
      )
    }

    const onMessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      switch (msg.type) {
        case 'stage':
          setModel((m) => ({ ...m, stage: msg.stage, backend: msg.backend ?? m.backend, dtype: msg.dtype ?? m.dtype, note: msg.note }))
          break
        case 'progress':
          setModel((m) => ({ ...m, files: msg.files, loaded: msg.loaded, total: msg.total }))
          break
        case 'ready':
          // Only remember a model once it has loaded, so a model that fails here isn't retried on every visit.
          try {
            localStorage.setItem(MODEL_STORAGE_KEY, msg.model)
          } catch {
            // Storage unavailable (private mode); the choice just won't be remembered.
          }
          setModel((m) => ({
            ...m,
            key: msg.model,
            stage: 'ready',
            backend: msg.backend,
            dtype: msg.dtype,
            loadMs: msg.loadMs,
            fallbackReason: msg.fallbackReason,
            note: undefined,
          }))
          // Process every image that has no result for this model yet (new uploads, or all after a switch).
          readyModel.current = msg.model
          enqueue(itemsRef.current, msg.model)
          setModelLoads((all) => [
            ...all,
            {
              model: msg.model,
              modelId: MODELS[msg.model].id,
              status: 'ready',
              backend: msg.backend,
              dtype: msg.dtype,
              loadMs: Math.round(msg.loadMs),
              fallbackReason: msg.fallbackReason,
              at: iso(Date.now()),
            },
          ])
          break
        case 'result': {
          const { id, backend, dtype, mask, width, height, timings } = msg
          updateResult(id, { status: 'done', backend, mask, width, height, timings })
          updateRun(id, {
            status: 'done',
            backend,
            dtype,
            width,
            height,
            timingsMs: {
              preprocessMs: Math.round(timings.preprocessMs),
              inferenceMs: Math.round(timings.inferenceMs),
              postprocessMs: Math.round(timings.postprocessMs),
              totalMs: Math.round(timings.totalMs),
            },
            startedAt: iso(msg.startedAt),
            finishedAt: iso(msg.finishedAt),
          })
          break
        }
        case 'cache':
          setCache(msg.info)
          break
        case 'gpu':
          setGpu(msg.info)
          break
        case 'needsDownload':
          setModel((m) => ({ ...m, key: msg.model, stage: 'not-downloaded', backend: msg.backend, dtype: msg.dtype }))
          break
        case 'restartOnCpu':
          gpuFailed.current.set(msg.model, msg.reason)
          setModelLoads((all) => [
            ...all,
            { model: msg.model, modelId: MODELS[msg.model].id, status: 'gpu-failed', backend: 'webgpu', error: msg.reason, at: iso(Date.now()) },
          ])
          startWorker().postMessage({
            type: 'load',
            model: msg.model,
            download: msg.download,
            forceCpu: true,
            fallbackReason: msg.reason,
          } satisfies WorkerRequest)
          break
        case 'error':
          if (msg.id !== undefined) {
            updateResult(msg.id, { status: 'error', error: msg.message })
            updateRun(msg.id, { status: 'error', error: msg.message, finishedAt: iso(msg.finishedAt ?? Date.now()) })
          } else {
            const key = loadingModel.current
            setModel((m) => ({ ...m, stage: 'error', error: msg.message }))
            setModelLoads((all) => [
              ...all,
              { model: key, modelId: MODELS[key].id, status: 'error', error: msg.message, at: iso(Date.now()) },
            ])
          }
          break
      }
    }

    // Load the last-used model only if it is already cached; otherwise the user is asked to download it.
    startWorker().postMessage({ type: 'load', model: loadingModel.current, download: false, forceCpu } satisfies WorkerRequest)
    return () => workerRef.current?.terminate()
  }, [enqueue])

  /** Switches to `key`. Without `download`, only cached files are used. */
  const loadModel = useCallback((key: ModelKey, download: boolean) => {
    loadingModel.current = key
    readyModel.current = null
    setModel(initialModelState(key))
    // Ask the browser to keep downloaded models instead of clearing them when it frees up space.
    // Chrome decides silently (no prompt); the download works either way.
    if (download) navigator.storage?.persist?.().catch(() => {})
    const gpuFailure = gpuFailed.current.get(key)
    workerRef.current?.postMessage({
      type: 'load',
      model: key,
      download,
      forceCpu: forceCpu || gpuFailure !== undefined,
      fallbackReason: gpuFailure,
    } satisfies WorkerRequest)
  }, [])

  const deleteCache = (key: ModelKey) => {
    workerRef.current?.postMessage({ type: 'deleteCache', model: key } satisfies WorkerRequest)
  }

  const addFiles = useCallback((files: File[]) => {
    const images = files.filter((f) => f.type.startsWith('image/'))
    if (images.length === 0) return
    const added: Item[] = images.map((file) => ({
      id: nextItemId.current++,
      name: file.name || 'pasted-image.png',
      blob: file,
      originalUrl: URL.createObjectURL(file),
      results: {},
    }))
    setItems((all) => [...all, ...added])
    setSelectedId(added[0].id)
    if (readyModel.current) enqueue(added, readyModel.current)
  }, [enqueue])

  const removeItem = (id: number) => {
    const index = items.findIndex((item) => item.id === id)
    if (index === -1) return
    const item = items[index]
    for (const result of Object.values(item.results)) {
      if (result?.status === 'queued') workerRef.current?.postMessage({ type: 'cancel', id: result.runId } satisfies WorkerRequest)
    }
    URL.revokeObjectURL(item.originalUrl)
    setRuns((all) =>
      all.map((r) => (r.imageId === id ? { ...r, imageRemoved: true, status: r.status === 'queued' ? 'cancelled' : r.status } : r)),
    )
    const rest = items.filter((other) => other.id !== id)
    setItems(rest)
    if (selectedId === id) setSelectedId(rest[Math.min(index, rest.length - 1)]?.id ?? null)
  }

  const resultFor = (item: Item) => item.results[model.key]

  const downloadItem = async (item: Item) => {
    const r = resultFor(item)
    if (r?.status !== 'done' || !r.mask || !r.width || !r.height) return
    saveFile(await renderCutout(item.blob, r.mask, r.width, r.height, cleanup, mainOnly), cutoutFileName(item.name))
  }

  const downloadAll = async () => {
    const ready = items.filter((item) => resultFor(item)?.status === 'done')
    if (ready.length === 0) return
    try {
      const files = []
      for (const [i, item] of ready.entries()) {
        setZipProgress(`Preparing ${i + 1}/${ready.length}…`)
        const r = resultFor(item)!
        files.push({ name: cutoutFileName(item.name), blob: await renderCutout(item.blob, r.mask!, r.width!, r.height!, cleanup, mainOnly) })
      }
      setZipProgress('Zipping…')
      saveFile(await zipFiles(files), `background-removed_${model.key}_${timestamp()}.zip`)
    } finally {
      setZipProgress(null)
    }
  }

  const exportLog = () => {
    const log = buildLog({ gpu, settings: { cleanupPercent: cleanup, keepMainSubjectOnly: mainOnly }, modelLoads, runs })
    saveFile(new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' }), `background-removal-log_${timestamp()}.json`)
  }

  // Allow pasting images from the clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => addFiles([...(e.clipboardData?.files ?? [])])
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [addFiles])

  const loadSample = async () => {
    const blob = await (await fetch(SAMPLE_URL)).blob()
    addFiles([new File([blob], 'sample.jpg', { type: blob.type })])
  }

  const selected = items.find((item) => item.id === selectedId) ?? null
  const queuedIds = items.filter((item) => resultFor(item)?.status === 'queued').map((item) => item.id)

  return (
    <main className="page">
      <header className="hero">
        <h1>On-device background removal</h1>
        <p>
          Runs entirely in your browser with Transformers.js. Your images never leave this device.
        </p>
      </header>

      <ModelPanel
        model={model}
        cache={cache}
        gpu={gpu}
        onSelect={(key) => loadModel(key, false)}
        onDownload={(key) => loadModel(key, true)}
        onDeleteCache={deleteCache}
      />

      <section className="card">
        <h2>1. Upload images</h2>
        <UploadZone modelReady={model.stage === 'ready'} onFiles={addFiles} />
        <div className="upload-actions">
          <button type="button" className="link" onClick={loadSample}>
            or try a sample image
          </button>
          <span className="hint">You can also paste images (Ctrl/⌘ + V)</span>
        </div>
      </section>

      <ItemList
        items={items}
        selectedId={selectedId}
        modelKey={model.key}
        modelStage={model.stage}
        queuedIds={queuedIds}
        runCount={runs.length}
        zipProgress={zipProgress}
        onSelect={setSelectedId}
        onRemove={removeItem}
        onDownload={downloadItem}
        onDownloadAll={downloadAll}
        onExportLog={exportLog}
      />

      {selected && (
        <ResultView
          key={`${selected.id}:${model.key}`}
          item={selected}
          result={resultFor(selected)}
          modelStage={model.stage}
          queuePosition={queuedIds.indexOf(selected.id)}
          cleanup={cleanup}
          onCleanupChange={setCleanup}
          mainOnly={mainOnly}
          onMainOnlyChange={setMainOnly}
          onDownload={() => downloadItem(selected)}
        />
      )}

      <footer className="footer">
        Model: <a href={`https://huggingface.co/${MODELS[model.key].id}`}>{MODELS[model.key].id}</a> (
        {MODELS[model.key].license} license) · Powered by{' '}
        <a href="https://huggingface.co/docs/transformers.js">Transformers.js</a>
      </footer>
    </main>
  )
}

function ModelPanel({
  model,
  cache,
  gpu,
  onSelect,
  onDownload,
  onDeleteCache,
}: {
  model: ModelState
  cache: CacheInfo | null
  gpu: GpuInfo | null
  onSelect: (key: ModelKey) => void
  onDownload: (key: ModelKey) => void
  onDeleteCache: (key: ModelKey) => void
}) {
  const busy = model.stage !== 'ready' && model.stage !== 'error' && model.stage !== 'not-downloaded'
  const pct = model.total > 0 ? Math.min(100, (model.loaded / model.total) * 100) : 0
  const backendLabel = model.backend === 'webgpu' ? 'WebGPU (GPU)' : model.backend === 'wasm' ? 'WASM (CPU)' : '—'
  const activeSpec = MODELS[model.key]
  const activeSizeMb = gpu ? plannedSizeMb(activeSpec, gpu, forceCpu) : undefined

  const status = {
    detecting: 'Checking for WebGPU support…',
    'not-downloaded': `${activeSpec.label} isn't downloaded yet.`,
    downloading: model.total > 0 ? `Downloading model… ${formatBytes(model.loaded)} / ${formatBytes(model.total)}` : 'Preparing download…',
    initializing: `${model.note ?? 'Initializing model'} on ${backendLabel}…`,
    ready: `Model ready in ${formatMs(model.loadMs ?? 0)}`,
    error: 'Failed to load the model',
  }[model.stage]

  const badge = { ready: 'Loaded', error: 'Error', 'not-downloaded': 'Not downloaded' }[model.stage as string] ?? 'Loading'
  const barPct = model.stage === 'ready' || model.stage === 'initializing' ? 100 : pct
  const indeterminate = model.stage === 'detecting' || model.stage === 'initializing' || (model.stage === 'downloading' && model.total === 0)

  return (
    <section className="card model">
      <div className="model-head">
        <h2>Model</h2>
        <span className={`badge ${model.stage}`}>{badge}</span>
      </div>

      <div className="model-options" role="radiogroup" aria-label="Model">
        {Object.values(MODELS).map((spec) => {
          const active = spec.key === model.key
          const cached = cache?.[spec.key]
          const cpuMb = spec.dtype.wasm && spec.sizeMb[spec.dtype.wasm]
          const unsupported = gpu && unsupportedReason(spec, gpu, forceCpu)
          const sizeMb = gpu ? plannedSizeMb(spec, gpu, forceCpu) : undefined
          const downloaded = Boolean(cached?.ready)
          const cacheText = downloaded
            ? `Downloaded${cached?.bytes ? ` · ${formatBytes(cached.bytes)}` : ''}`
            : active && model.stage === 'ready'
              ? 'Files deleted · loaded until you reload'
              : cached?.files
                ? `Partly downloaded · ${formatBytes(cached.bytes)}`
                : 'Not downloaded'
          return (
            <div key={spec.key} className={`model-option ${active ? 'active' : ''}`}>
              <button
                type="button"
                role="radio"
                aria-checked={active}
                className="model-pick"
                disabled={busy || active || Boolean(unsupported)}
                onClick={() => onSelect(spec.key)}
              >
                <span className="model-name">
                  {spec.label}
                  <span className={`license ${spec.commercial ? '' : 'restricted'}`}>{spec.license}</span>
                </span>
                <span className="hint">{spec.description}</span>
                <span className="hint">
                  Download: ~{spec.sizeMb[spec.dtype.webgpu.fp16]} MB on GPU
                  {cpuMb ? `, ~${cpuMb} MB on CPU` : ' (GPU only)'}
                </span>
                {unsupported && <span className="unsupported">Not available on this device. {unsupported}</span>}
              </button>
              <div className="model-cache">
                <span className={`cache-state ${downloaded ? 'cached' : ''}`}>{cacheText}</span>
                <span className="cache-actions">
                  {!downloaded && !unsupported && (
                    <button type="button" className="link" disabled={busy} onClick={() => onDownload(spec.key)}>
                      Download{sizeMb ? ` (~${sizeMb} MB)` : ''}
                    </button>
                  )}
                  {Boolean(cached?.files) && (
                    <button type="button" className="link danger" disabled={busy} onClick={() => onDeleteCache(spec.key)}>
                      Delete
                    </button>
                  )}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="model-meta">
        <div>
          <span className="label">Backend</span>
          <span>{backendLabel}</span>
        </div>
        <div>
          <span className="label">Precision</span>
          <span>{model.dtype ?? '—'}</span>
        </div>
        <div>
          <span className="label">Size</span>
          <span>{model.total > 0 ? formatBytes(model.total) : activeSizeMb ? `~${activeSizeMb} MB` : '—'}</span>
        </div>
      </div>

      {model.stage === 'not-downloaded' ? (
        <div className="download-cta">
          <p>
            <strong>{status}</strong> It's about {activeSizeMb ?? '?'} MB and is saved in your browser, so you only download
            it once. Images you add wait until it's ready.
          </p>
          <button type="button" className="button" onClick={() => onDownload(model.key)}>
            Download {activeSpec.label}
          </button>
        </div>
      ) : (
        <>
          <div className={`progress ${indeterminate ? 'indeterminate' : ''} ${model.stage}`}>
            <div className="progress-bar" style={{ width: `${barPct}%` }} />
          </div>
          <p className="status">
            {status}
            {model.stage === 'downloading' && model.total > 0 && <strong> {pct.toFixed(0)}%</strong>}
          </p>
        </>
      )}

      {model.fallbackReason && model.backend === 'wasm' && (
        <p className="warn">Using CPU fallback — {model.fallbackReason}. Processing will be slower.</p>
      )}
      {model.error && <p className="error">{model.error}</p>}

      {model.files.length > 0 && model.stage !== 'ready' && model.stage !== 'not-downloaded' && (
        <ul className="files">
          {model.files.map((f) => (
            <li key={f.file}>
              <span className="file-name">{f.file}</span>
              <span className="file-size">
                {f.done ? '✓' : f.total ? `${((f.loaded / f.total) * 100).toFixed(0)}%` : '…'}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function UploadZone({ modelReady, onFiles }: { modelReady: boolean; onFiles: (files: File[]) => void }) {
  const [dragging, setDragging] = useState(false)

  return (
    <label
      className={`dropzone ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        onFiles([...e.dataTransfer.files])
      }}
    >
      <input
        type="file"
        accept="image/*"
        multiple
        onChange={(e) => {
          onFiles([...(e.target.files ?? [])])
          e.target.value = ''
        }}
      />
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
        <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <strong>Drop images here or click to browse</strong>
      <span className="hint">
        PNG, JPG, WebP · select several at once{modelReady ? '' : ' · they start processing once the model has loaded'}
      </span>
    </label>
  )
}

function ItemList({
  items,
  selectedId,
  modelKey,
  modelStage,
  queuedIds,
  runCount,
  zipProgress,
  onSelect,
  onRemove,
  onDownload,
  onDownloadAll,
  onExportLog,
}: {
  items: Item[]
  selectedId: number | null
  modelKey: ModelKey
  modelStage: LoadStage
  queuedIds: number[]
  runCount: number
  zipProgress: string | null
  onSelect: (id: number) => void
  onRemove: (id: number) => void
  onDownload: (item: Item) => void
  onDownloadAll: () => void
  onExportLog: () => void
}) {
  const doneCount = items.filter((item) => item.results[modelKey]?.status === 'done').length
  if (items.length === 0 && runCount === 0) return null

  return (
    <section className="card">
      <div className="list-head">
        <h2>2. Images ({items.length})</h2>
        <div className="list-actions">
          <button type="button" className="button secondary" onClick={onExportLog} disabled={runCount === 0}>
            Export log (JSON)
          </button>
          <button type="button" className="button" onClick={onDownloadAll} disabled={doneCount === 0 || zipProgress !== null}>
            {zipProgress ?? `Download all (${doneCount}) as ZIP`}
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <p className="hint">All images were removed. The log still has their runs.</p>
      ) : (
        <ul className="items">
          {items.map((item) => {
            const result = item.results[modelKey]
            const position = queuedIds.indexOf(item.id)
            const status =
              result?.status === 'done' && result.timings
                ? { cls: 'done', text: formatMs(result.timings.totalMs) }
                : result?.status === 'error'
                  ? { cls: 'error', text: 'Failed' }
                  : result?.status === 'queued'
                    ? { cls: 'queued', text: position === 0 ? 'Processing…' : `Queued #${position + 1}` }
                    : { cls: 'waiting', text: modelStage === 'error' ? 'Model failed' : 'Waiting for model' }
            return (
              <li key={item.id} className={`item ${item.id === selectedId ? 'selected' : ''}`}>
                <button
                  type="button"
                  className="item-select"
                  aria-pressed={item.id === selectedId}
                  onClick={() => onSelect(item.id)}
                >
                  <img className="thumb" src={item.originalUrl} alt="" />
                  <span className="item-name" title={item.name}>
                    {item.name}
                  </span>
                  <span className={`item-status ${status.cls}`}>{status.text}</span>
                </button>
                <div className="item-actions">
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => onDownload(item)}
                    disabled={result?.status !== 'done'}
                    aria-label={`Download ${item.name} without background`}
                    title="Download PNG"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                      <path d="M12 4v12m0 0-5-5m5 5 5-5M5 20h14" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    className="icon-button danger"
                    onClick={() => onRemove(item.id)}
                    aria-label={`Remove ${item.name}`}
                    title="Remove"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                      <path d="M6 6l12 12M18 6 6 18" strokeLinecap="round" />
                    </svg>
                  </button>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

function ResultView({
  item,
  result,
  modelStage,
  queuePosition,
  cleanup,
  onCleanupChange,
  mainOnly,
  onMainOnlyChange,
  onDownload,
}: {
  item: Item
  result?: RunResult
  modelStage: LoadStage
  queuePosition: number
  cleanup: number
  onCleanupChange: (value: number) => void
  mainOnly: boolean
  onMainOnlyChange: (value: boolean) => void
  onDownload: () => void
}) {
  const [pixels, setPixels] = useState<{ data: ImageData; decodeMs: number } | null>(null)
  const [composeMs, setComposeMs] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mask = result?.mask
  const width = result?.width
  const height = result?.height

  // Decode the original once the mask arrives; the cutout is re-composited from it on every slider change.
  useEffect(() => {
    if (!mask || !width || !height) return
    let cancelled = false
    const start = performance.now()
    decodePixels(item.blob, width, height).then((data) => {
      if (!cancelled) setPixels({ data, decodeMs: performance.now() - start })
    })
    return () => {
      cancelled = true
    }
  }, [item.blob, mask, width, height])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !pixels || !mask) return
    const start = performance.now()
    let cutout = applyMask(pixels.data, mask, alphaCurve(cleanup / 100))
    if (mainOnly) cutout = keepMainSubject(cutout)
    canvas.width = cutout.width
    canvas.height = cutout.height
    canvas.getContext('2d')!.putImageData(cutout, 0, 0)
    setComposeMs(pixels.decodeMs + performance.now() - start)
  }, [pixels, mask, cleanup, mainOnly])

  const done = Boolean(mask && pixels)
  const waitingText =
    result?.status === 'queued'
      ? queuePosition > 0
        ? `Queued, ${queuePosition} image${queuePosition > 1 ? 's' : ''} ahead…`
        : 'Removing background…'
      : modelStage === 'error'
        ? 'The model failed to load. Pick another model above.'
        : 'Waiting for the model to load…'

  return (
    <section className="card">
      <div className="result-head">
        <h2 className="result-title">
          3. Result <span className="result-name">{item.name}</span>
        </h2>
        {done && (
          <button type="button" className="button" onClick={onDownload}>
            Download PNG
          </button>
        )}
      </div>

      <ResultStats result={result} composeMs={done ? composeMs : undefined} />

      {result?.status === 'error' ? (
        <p className="error">Failed to remove background: {result.error}</p>
      ) : mask ? (
        <>
          <label className="cleanup">
            <span className="cleanup-head">
              <span>Background cleanup</span>
              <strong>{cleanup}%</strong>
            </span>
            <input type="range" min={0} max={60} value={cleanup} onChange={(e) => onCleanupChange(Number(e.target.value))} />
            <span className="hint">
              Removes faint leftovers from busy backgrounds (text, patterns). 0% shows the raw model output; raise it if
              traces remain, lower it if fine edges like hair get cut. Applies to all images and downloads.
            </span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={mainOnly} onChange={(e) => onMainOnlyChange(e.target.checked)} />
            <span>
              <span>Keep main subject only</span>
              <span className="hint">
                Removes separate blobs much smaller than the main subject, like promo text, logos or props beside a
                product. Turn off if parts of your subject are separated from each other.
              </span>
            </span>
          </label>
          <BeforeAfter before={item.originalUrl} canvasRef={canvasRef} width={width} height={height} />
        </>
      ) : (
        <div className="processing">
          <img src={item.originalUrl} alt="Original" />
          <div className="processing-overlay">
            {modelStage !== 'error' && <span className="spinner" />} {waitingText}
          </div>
        </div>
      )}
    </section>
  )
}

function ResultStats({ result, composeMs }: { result?: RunResult; composeMs?: number }) {
  const t =
    result?.timings && composeMs !== undefined
      ? { ...result.timings, postprocessMs: result.timings.postprocessMs + composeMs, totalMs: result.timings.totalMs + composeMs }
      : undefined
  return (
    <div className="stats">
      <div className="stat main">
        <span className="label">Background removed in</span>
        <span className="value">
          {t ? formatMs(t.totalMs) : result?.status === 'error' || !result ? '—' : <span className="spinner" />}
        </span>
        <span className="hint">
          {result ? MODELS[result.model].label : 'Not processed yet'}
          {result?.backend && ` on ${result.backend === 'webgpu' ? 'GPU (WebGPU)' : 'CPU (WASM)'}`}
        </span>
      </div>
      <div className="stat">
        <span className="label">Pre-process</span>
        <span className="value small">{t ? formatMs(t.preprocessMs) : '—'}</span>
      </div>
      <div className="stat">
        <span className="label">Inference</span>
        <span className="value small">{t ? formatMs(t.inferenceMs) : '—'}</span>
      </div>
      <div className="stat">
        <span className="label">Post-process</span>
        <span className="value small">{t ? formatMs(t.postprocessMs) : '—'}</span>
      </div>
      <div className="stat">
        <span className="label">Image size</span>
        <span className="value small">{result?.width ? `${result.width}×${result.height}` : '—'}</span>
      </div>
    </div>
  )
}

function BeforeAfter({
  before,
  canvasRef,
  width,
  height,
}: {
  before: string
  canvasRef: RefObject<HTMLCanvasElement | null>
  width?: number
  height?: number
}) {
  const [pos, setPos] = useState(50)
  const ref = useRef<HTMLDivElement>(null)

  const update = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    setPos(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)))
  }

  return (
    <div className="compare-wrap">
      <div
        ref={ref}
        className="compare"
        style={
          width && height
            ? { aspectRatio: `${width} / ${height}`, width: `min(100%, calc(70vh * ${width / height}))` }
            : undefined
        }
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          update(e.clientX)
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) update(e.clientX)
        }}
      >
        <canvas ref={canvasRef} className="after checkerboard" aria-label="Background removed" />
        <img className="before" src={before} alt="Original" draggable={false} style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
        <div className="divider" style={{ left: `${pos}%` }}>
          <span className="handle" />
        </div>
        <span className="tag left">Before</span>
        <span className="tag right">After</span>
      </div>
      <input
        className="slider"
        type="range"
        min={0}
        max={100}
        value={pos}
        aria-label="Before / after comparison"
        onChange={(e) => setPos(Number(e.target.value))}
      />
    </div>
  )
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

function formatMs(ms: number) {
  return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`
}
