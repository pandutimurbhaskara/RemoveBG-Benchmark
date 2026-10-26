import { zipSync } from 'fflate'
import type { Backend, Timings } from './messages'
import type { GpuInfo, ModelKey } from './models'

/** Saves a blob as a file through a temporary download link. */
export function saveFile(blob: Blob, fileName: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = fileName
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 0)
}

/** "photo.jpg" -> "photo-no-bg.png" */
export const cutoutFileName = (name: string) => name.replace(/\.[^.]+$/, '') + '-no-bg.png'

/** Packs files into an uncompressed ZIP (PNGs are already compressed). Duplicate names get a counter. */
export async function zipFiles(files: { name: string; blob: Blob }[]): Promise<Blob> {
  const entries: Record<string, Uint8Array> = {}
  for (const { name, blob } of files) {
    let unique = name
    for (let n = 2; unique in entries; n++) unique = name.replace(/(\.[^.]+)?$/, `-${n}$1`)
    entries[unique] = new Uint8Array(await blob.arrayBuffer())
  }
  return new Blob([zipSync(entries, { level: 0 })], { type: 'application/zip' })
}

/** Local date-time stamp for file names, e.g. 2026-10-26_14-05-09. */
export function timestamp(date = new Date()) {
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}_` +
    `${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`
  )
}

export interface ModelLoadLog {
  model: ModelKey
  modelId: string
  status: 'ready' | 'error' | 'gpu-failed'
  backend?: Backend
  dtype?: string
  loadMs?: number
  fallbackReason?: string
  error?: string
  at: string
}

export interface RunLog {
  runId: number
  imageId: number
  file: { name: string; type: string; bytes: number }
  model: ModelKey
  modelId: string
  status: 'queued' | 'done' | 'error' | 'cancelled'
  imageRemoved: boolean
  backend?: Backend
  dtype?: string
  width?: number
  height?: number
  timingsMs?: Timings
  error?: string
  queuedAt: string
  startedAt?: string
  finishedAt?: string
}

export function buildLog({
  gpu,
  settings,
  modelLoads,
  runs,
}: {
  gpu: GpuInfo | null
  settings: { cleanupPercent: number; keepMainSubjectOnly: boolean }
  modelLoads: ModelLoadLog[]
  runs: RunLog[]
}) {
  const nav = navigator as Navigator & { deviceMemory?: number }
  const done = runs.filter((r) => r.status === 'done' && r.timingsMs)
  return {
    app: 'on-device-background-removal',
    exportedAt: new Date().toISOString(),
    environment: {
      userAgent: nav.userAgent,
      hardwareConcurrency: nav.hardwareConcurrency,
      deviceMemoryGb: nav.deviceMemory ?? null,
      crossOriginIsolated: self.crossOriginIsolated,
      webgpu: gpu,
    },
    settings,
    notes: [
      'timingsMs are measured inside the worker (decode + preprocess, inference, mask resize).',
      'They exclude compositing with the cleanup settings, which happens when the result is shown or downloaded.',
    ],
    summary: {
      runs: runs.length,
      done: done.length,
      errors: runs.filter((r) => r.status === 'error').length,
      averageTotalMsByModel: Object.fromEntries(
        [...new Set(done.map((r) => r.model))].map((model) => {
          const totals = done.filter((r) => r.model === model).map((r) => r.timingsMs!.totalMs)
          return [model, Math.round(totals.reduce((a, b) => a + b, 0) / totals.length)]
        }),
      ),
    },
    modelLoads,
    runs,
  }
}
