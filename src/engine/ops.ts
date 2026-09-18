// Core tensor operations and post-transforms used by the reference engine.

import type { Tensor, DataType, PostTransform } from '../ir.js';

// ── TensorData — internal engine representation ───────────────────────────────

export type NumericArray = Float64Array | Float32Array | Int32Array;

export interface TensorData {
  dtype: DataType;
  shape: number[];   // e.g. [N, D]; first dim is always batch N
  data: Float64Array | string[] | boolean[];
}

export function tensorRows(td: TensorData): number {
  return td.shape[0] ?? 0;
}

export function tensorCols(td: TensorData): number {
  if (td.shape.length < 2) return 1;
  return td.shape.slice(1).reduce((a, b) => a * b, 1);
}

// ── Tensor → TensorData conversion ───────────────────────────────────────────

export function tensorToData(tensor: Tensor | number[]): TensorData {
  // Some JSON serializations emit plain arrays instead of Tensor objects (e.g. split_threshold, leaf_value).
  if (Array.isArray(tensor)) {
    return { dtype: 'FLOAT64', shape: [tensor.length], data: new Float64Array(tensor) };
  }

  const dtype = tensor.type?.dtype ?? 'FLOAT64';
  const shape = tensor.type?.shape ?? [];

  if (dtype === 'STRING') {
    const data = tensor.string_data ?? [];
    return { dtype, shape, data };
  }

  if (dtype === 'BOOL') {
    const data = tensor.bool_data ?? [];
    return { dtype, shape, data };
  }

  if (tensor.raw_data) {
    return { dtype, shape, data: decodeRawData(tensor.raw_data, dtype, shape) };
  }

  let nums: number[];
  if (tensor.float32_data) nums = tensor.float32_data;
  else if (tensor.float64_data) nums = tensor.float64_data;
  else if (tensor.int32_data) nums = tensor.int32_data;
  else if (tensor.int64_data) nums = tensor.int64_data;
  else nums = [];

  return { dtype, shape, data: new Float64Array(nums) };
}

function decodeRawData(b64: string, dtype: DataType, shape: number[]): Float64Array {
  const bytes = base64ToBytes(b64);
  const n = shape.reduce((a, b) => a * b, 1);
  const out = new Float64Array(n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  switch (dtype) {
    case 'FLOAT32':
      for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
      break;
    case 'FLOAT64':
      for (let i = 0; i < n; i++) out[i] = view.getFloat64(i * 8, true);
      break;
    case 'INT32':
      for (let i = 0; i < n; i++) out[i] = view.getInt32(i * 4, true);
      break;
    case 'INT64':
      for (let i = 0; i < n; i++) out[i] = Number(view.getBigInt64(i * 8, true));
      break;
    case 'INT8':
      for (let i = 0; i < n; i++) out[i] = view.getInt8(i);
      break;
    case 'UINT8':
      for (let i = 0; i < n; i++) out[i] = view.getUint8(i);
      break;
    case 'INT16':
      for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true);
      break;
    case 'UINT16':
      for (let i = 0; i < n; i++) out[i] = view.getUint16(i * 2, true);
      break;
    case 'UINT32':
      for (let i = 0; i < n; i++) out[i] = view.getUint32(i * 4, true);
      break;
    default:
      // Unsupported dtype for raw binary decode — return zeros
      break;
  }
  return out;
}

function base64ToBytes(b64: string): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g = globalThis as any;
  if (typeof g['Buffer'] !== 'undefined') {
    return new Uint8Array(g['Buffer'].from(b64, 'base64') as ArrayBuffer);
  }
  const bin: string = g['atob'](b64) as string;
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// ── Post-transforms ───────────────────────────────────────────────────────────

export function applyPostTransform(
  scores: Float64Array,
  transform: PostTransform | undefined,
  outputSize: number,  // per-row output width (for softmax)
): Float64Array {
  switch (transform) {
    case undefined:
    case 'POST_TRANSFORM_UNSPECIFIED':
    case 'IDENTITY':
      return scores;

    case 'SIGMOID': {
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) out[i] = 1 / (1 + Math.exp(-scores[i]));
      return out;
    }

    case 'SIGMOID_BINARY': {
      // Sigmoid then expand to 2-column [1-p, p] per row
      const N = scores.length / outputSize;
      const out = new Float64Array(N * 2);
      for (let row = 0; row < N; row++) {
        const p = 1 / (1 + Math.exp(-scores[row * outputSize]));
        out[row * 2] = 1 - p;
        out[row * 2 + 1] = p;
      }
      return out;
    }

    case 'SOFTMAX': {
      const out = new Float64Array(scores.length);
      const N = scores.length / outputSize;
      for (let row = 0; row < N; row++) {
        const offset = row * outputSize;
        let maxVal = -Infinity;
        for (let k = 0; k < outputSize; k++) maxVal = Math.max(maxVal, scores[offset + k]);
        let sum = 0;
        for (let k = 0; k < outputSize; k++) {
          out[offset + k] = Math.exp(scores[offset + k] - maxVal);
          sum += out[offset + k];
        }
        for (let k = 0; k < outputSize; k++) out[offset + k] /= sum;
      }
      return out;
    }

    case 'EXP': {
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) out[i] = Math.exp(scores[i]);
      return out;
    }

    case 'LOGIT': {
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) {
        const p = scores[i];
        out[i] = Math.log(p / (1 - p));
      }
      return out;
    }

    case 'PROBIT': {
      // Rational approximation for inverse normal CDF
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) out[i] = probitApprox(scores[i]);
      return out;
    }

    case 'CLOGLOG': {
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) out[i] = 1 - Math.exp(-Math.exp(scores[i]));
      return out;
    }

    case 'LOGLOG': {
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) out[i] = Math.exp(-Math.exp(-scores[i]));
      return out;
    }

    case 'CAUCHIT': {
      const out = new Float64Array(scores.length);
      for (let i = 0; i < scores.length; i++) out[i] = 0.5 + (1 / Math.PI) * Math.atan(scores[i]);
      return out;
    }

    default:
      return scores;
  }
}

export function applyActivation(
  x: Float64Array,
  activation: string,
  outputSize: number,
): Float64Array {
  switch (activation) {
    case 'IDENTITY': return x;
    case 'LOGISTIC': {
      const out = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
      return out;
    }
    case 'TANH': {
      const out = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
      return out;
    }
    case 'RELU': {
      const out = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) out[i] = Math.max(0, x[i]);
      return out;
    }
    case 'SOFTMAX':
      return applyPostTransform(x, 'SOFTMAX', outputSize);
    default:
      return x;
  }
}

// ── Matrix multiply: out[N, outD] = in[N, inD] @ W[outD, inD]^T + b[outD] ───

export function matmulBias(
  input: Float64Array, N: number, inD: number,
  weights: Float64Array, outD: number,
  bias: Float64Array | null,
): Float64Array {
  const out = new Float64Array(N * outD);
  for (let row = 0; row < N; row++) {
    for (let j = 0; j < outD; j++) {
      let acc = bias ? bias[j] : 0;
      for (let k = 0; k < inD; k++) {
        acc += input[row * inD + k] * weights[j * inD + k];
      }
      out[row * outD + j] = acc;
    }
  }
  return out;
}

// ── Probit approximation ──────────────────────────────────────────────────────

function probitApprox(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  // Beasley-Springer-Moro algorithm approximation
  const a = [2.515517, 0.802853, 0.010328];
  const b = [1.432788, 0.189269, 0.001308];
  const sign = p < 0.5 ? -1 : 1;
  const q = Math.min(p, 1 - p);
  const t = Math.sqrt(-2 * Math.log(q));
  const num = a[0] + a[1] * t + a[2] * t * t;
  const den = 1 + b[0] * t + b[1] * t * t + b[2] * t * t * t;
  return sign * (t - num / den);
}
