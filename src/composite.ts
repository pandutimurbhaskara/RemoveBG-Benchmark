/**
 * Builds a lookup table that turns the model's soft mask into the final alpha channel.
 *
 * The model is unsure about busy backgrounds (text, patterns, props) and gives them a low but
 * non-zero alpha, which shows up as faint "ghost" traces. Cleanup raises the floor below which
 * pixels become fully transparent, and stretches the rest so edges (hair, fan grills) stay soft.
 *
 * @param strength 0 = raw model output, higher = more aggressive cleanup (0..1).
 */
export function alphaCurve(strength: number): Uint8ClampedArray {
  const lo = strength * 255
  const hi = lo + (255 - lo) * (1 - strength)
  const lut = new Uint8ClampedArray(256)
  for (let v = 0; v < 256; v++) {
    lut[v] = hi > lo ? ((v - lo) / (hi - lo)) * 255 : v >= lo ? 255 : 0
  }
  return lut
}

/** Decodes an image blob into RGBA pixels at the given size. */
export async function decodePixels(blob: Blob, width: number, height: number): Promise<ImageData> {
  const bitmap = await createImageBitmap(blob)
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(bitmap, 0, 0, width, height)
  bitmap.close()
  return ctx.getImageData(0, 0, width, height)
}

/** Returns a copy of `pixels` whose alpha channel comes from `mask` passed through `lut`. */
export function applyMask(pixels: ImageData, mask: Uint8ClampedArray, lut: Uint8ClampedArray): ImageData {
  const out = new Uint8ClampedArray(pixels.data)
  for (let i = 0, p = 3; i < mask.length; i++, p += 4) {
    out[p] = lut[mask[i]]
  }
  return new ImageData(out, pixels.width, pixels.height)
}

/**
 * Clears visible regions that aren't connected to the main subject, such as promo text, logos or
 * props placed beside a product. Regions smaller than `minRatio` × the largest one are removed, so
 * several similarly sized products in one photo are all kept. Mutates and returns `image`.
 */
export function keepMainSubject(image: ImageData, minRatio = 0.25): ImageData {
  const { width, height, data } = image
  const count = width * height
  const labels = new Int32Array(count) // 0 = transparent or not yet visited
  const sizes = [0]
  const stack = new Int32Array(count)

  // Flood-fill every visible (alpha > 0) region with 8-connectivity.
  for (let start = 0; start < count; start++) {
    if (labels[start] || data[start * 4 + 3] === 0) continue
    const label = sizes.length
    let size = 0
    let top = 0
    stack[top++] = start
    labels[start] = label
    while (top) {
      const p = stack[--top]
      size++
      const x = p % width
      const y = (p - x) / width
      for (let ny = Math.max(0, y - 1); ny <= Math.min(height - 1, y + 1); ny++) {
        for (let nx = Math.max(0, x - 1); nx <= Math.min(width - 1, x + 1); nx++) {
          const q = ny * width + nx
          if (!labels[q] && data[q * 4 + 3] > 0) {
            labels[q] = label
            stack[top++] = q
          }
        }
      }
    }
    sizes.push(size)
  }

  let largest = 0
  for (const size of sizes) largest = Math.max(largest, size)
  const minSize = largest * minRatio
  for (let p = 0; p < count; p++) {
    if (labels[p] && sizes[labels[p]] < minSize) data[p * 4 + 3] = 0
  }
  return image
}

/** Renders the final cutout PNG for an image and its mask with the given cleanup settings. */
export async function renderCutout(
  blob: Blob,
  mask: Uint8ClampedArray,
  width: number,
  height: number,
  cleanupPercent: number,
  mainOnly: boolean,
): Promise<Blob> {
  const pixels = await decodePixels(blob, width, height)
  let cutout = applyMask(pixels, mask, alphaCurve(cleanupPercent / 100))
  if (mainOnly) cutout = keepMainSubject(cutout)
  const canvas = new OffscreenCanvas(width, height)
  canvas.getContext('2d')!.putImageData(cutout, 0, 0)
  return canvas.convertToBlob({ type: 'image/png' })
}
