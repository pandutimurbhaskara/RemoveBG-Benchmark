export type ModelKey =
  | 'birefnet-lite-512'
  | 'birefnet-lite-1024-webgpu'
  | 'isnet-general'
  | 'ben2'
  | 'modnet'
  | 'rmbg-1.4'
  | 'mvanet'
  | 'inspyrenet'

export type Dtype = 'fp32' | 'fp16' | 'q8' | 'q4f16'

export interface ModelSpec {
  key: ModelKey
  id: string
  label: string
  description: string
  license: string
  /** Whether the license allows commercial use. */
  commercial: boolean
  /**
   * Precision per backend. WebGPU uses `fp16` when the GPU supports `shader-f16`, else `fallback`
   * (`null` when the model only ships fp16 weights). `wasm: null` means the model can't run on CPU.
   */
  dtype: { webgpu: { fp16: Dtype; fallback: Dtype | null }; wasm: Dtype | null }
  /** Approximate download size in MB per precision, shown before the model is downloaded. */
  sizeMb: Partial<Record<Dtype, number>>
  /**
   * Minimum WebGPU `maxStorageBuffersPerShaderStage` the model's kernels need (the spec default is 8,
   * Chrome allows 10 on Apple GPUs). Ops with many outputs, like a 32-way Split, can exceed it.
   */
  minStorageBuffersPerShaderStage?: number
  /** Overrides for config.json, for architectures transformers.js doesn't know. */
  config?: Record<string, unknown>
  /**
   * ONNX file name (without `.onnx`) when the repo doesn't follow the `model_<dtype>` naming.
   * transformers.js then loads this exact file, and `dtype` only describes the I/O.
   */
  modelFileName?: string
  /** Precision to display when `modelFileName` hides the real one. */
  precisionLabel?: string
}

export const MODELS: Record<ModelKey, ModelSpec> = {
  'birefnet-lite-512': {
    key: 'birefnet-lite-512',
    id: 'studioludens/birefnet-lite-512',
    label: 'BiRefNet lite 512',
    description: 'Clean cutouts on busy backgrounds. Runs on GPU and CPU.',
    license: 'MIT',
    commercial: true,
    dtype: { webgpu: { fp16: 'fp16', fallback: 'fp32' }, wasm: 'fp32' },
    sizeMb: { fp16: 99, fp32: 192 },
  },
  'birefnet-lite-1024-webgpu': {
    key: 'birefnet-lite-1024-webgpu',
    id: 'jiabins0303/birefnet-lite-1024-webgpu',
    label: 'BiRefNet lite 1024',
    description: 'Full-resolution BiRefNet, graph patched to run on WebGPU. Sharpest edges, slower.',
    license: 'MIT',
    commercial: true,
    // The file is already fp16 inside; `fp32` + an explicit file name stops transformers.js
    // from appending a dtype suffix. It needs shader-f16, hence no fp32 fallback.
    dtype: { webgpu: { fp16: 'fp32', fallback: null }, wasm: null },
    sizeMb: { fp32: 115 },
    minStorageBuffersPerShaderStage: 8,
    modelFileName: 'model_fp16',
    precisionLabel: 'fp16',
  },
  'isnet-general': {
    key: 'isnet-general',
    id: 'Ko033/isnet-general-use-onnx',
    label: 'IS-Net general',
    description: 'Same architecture and size as RMBG-1.4 (original DIS weights).',
    license: 'Apache-2.0',
    commercial: true,
    dtype: { webgpu: { fp16: 'fp16', fallback: null }, wasm: 'q8' },
    sizeMb: { fp16: 88, q8: 46 },
  },
  ben2: {
    key: 'ben2',
    id: 'onnx-community/BEN2-ONNX',
    label: 'BEN2',
    description: 'Most precise edges (hair, see-through parts). Very slow on CPU (~15 s).',
    license: 'MIT',
    commercial: true,
    dtype: { webgpu: { fp16: 'fp16', fallback: null }, wasm: 'fp16' },
    sizeMb: { fp16: 219 },
  },
  modnet: {
    key: 'modnet',
    id: 'Xenova/modnet',
    label: 'MODNet',
    description: 'Tiny and very fast. People and portraits only.',
    license: 'Apache-2.0',
    commercial: true,
    dtype: { webgpu: { fp16: 'fp16', fallback: 'fp32' }, wasm: 'q8' },
    sizeMb: { fp16: 13, fp32: 26, q8: 7 },
  },
  mvanet: {
    key: 'mvanet',
    id: 'onnx-community/MVANet-ONNX',
    label: 'MVANet',
    description: 'Precise, clean cutouts (CVPR 2024). ~3.5 s on GPU, very slow on CPU (~16 s).',
    license: 'MIT',
    commercial: true,
    // q4f16 looked identical to fp16 in testing (same region scores) at 40% of the download.
    dtype: { webgpu: { fp16: 'q4f16', fallback: 'fp32' }, wasm: 'q8' },
    sizeMb: { q4f16: 85, fp32: 422, q8: 141 },
  },
  inspyrenet: {
    key: 'inspyrenet',
    id: 'OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX',
    label: 'InSPyReNet',
    description: 'High-quality cutouts. ~4 s on GPU, large download, GPU only.',
    license: 'MIT',
    commercial: true,
    // fp16 on CPU returns an empty mask, and fp32 (395 MB) is too big for WASM memory.
    dtype: { webgpu: { fp16: 'fp16', fallback: 'fp32' }, wasm: null },
    sizeMb: { fp16: 199, fp32: 395 },
  },
  'rmbg-1.4': {
    key: 'rmbg-1.4',
    id: 'briaai/RMBG-1.4',
    label: 'RMBG-1.4',
    description: 'Fast. Can leave faint traces on busy backgrounds.',
    license: 'Non-commercial',
    commercial: false,
    dtype: { webgpu: { fp16: 'fp16', fallback: 'fp32' }, wasm: 'q8' },
    sizeMb: { fp16: 88, fp32: 176, q8: 44 },
    config: { model_type: 'custom' },
  },
}

export const DEFAULT_MODEL: ModelKey = 'birefnet-lite-512'

/** The precision to use on WebGPU, or null if the model's weights can't run on this GPU. */
export const webgpuDtype = (spec: ModelSpec, gpuFp16: boolean) => (gpuFp16 ? spec.dtype.webgpu.fp16 : spec.dtype.webgpu.fallback)

export const isModelKey = (value: unknown): value is ModelKey => typeof value === 'string' && value in MODELS

export type GpuInfo =
  | { available: true; fp16: boolean; maxStorageBuffersPerShaderStage: number }
  | { available: false; reason: string }

/** Why `spec` can't run on this GPU, or null if it can. */
export function gpuProblem(spec: ModelSpec, gpu: GpuInfo): string | null {
  if (!gpu.available) return gpu.reason
  if (!webgpuDtype(spec, gpu.fp16)) return 'this GPU has no fp16 support (shader-f16), which the model needs'
  const needed = spec.minStorageBuffersPerShaderStage ?? 0
  const allowed = gpu.maxStorageBuffersPerShaderStage
  if (allowed < needed) return `this GPU allows ${allowed} storage buffers per shader stage, the model needs ${needed}`
  return null
}

/** Why `spec` can't run on this device at all (neither GPU nor CPU), or null if it can. */
export function unsupportedReason(spec: ModelSpec, gpu: GpuInfo, forceCpu: boolean): string | null {
  if (spec.dtype.wasm) return null
  const problem = forceCpu ? 'CPU mode is forced' : gpuProblem(spec, gpu)
  return problem && `Needs WebGPU; ${problem}.`
}

const DTYPE_SUFFIX: Record<Dtype, string> = { fp32: '', fp16: '_fp16', q8: '_quantized', q4f16: '_q4f16' }

/** Path of the ONNX weights transformers.js loads for `dtype`, relative to the repo root. */
export const onnxFile = (spec: ModelSpec, dtype: Dtype) => `onnx/${spec.modelFileName ?? `model${DTYPE_SUFFIX[dtype]}`}.onnx`

/** The backend and precision `spec` will run with on this device, or null if it can't run here. */
export function plannedRun(spec: ModelSpec, gpu: GpuInfo, forceCpu: boolean): { backend: 'webgpu' | 'wasm'; dtype: Dtype } | null {
  if (!forceCpu && gpu.available && !gpuProblem(spec, gpu)) return { backend: 'webgpu', dtype: webgpuDtype(spec, gpu.fp16)! }
  return spec.dtype.wasm ? { backend: 'wasm', dtype: spec.dtype.wasm } : null
}

/** Approximate download size in MB for the planned run. */
export const plannedSizeMb = (spec: ModelSpec, gpu: GpuInfo, forceCpu: boolean) => {
  const plan = plannedRun(spec, gpu, forceCpu)
  return plan ? spec.sizeMb[plan.dtype] : undefined
}
