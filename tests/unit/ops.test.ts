// Unit tests for engine/ops.ts — pure numeric functions with no I/O.

import { describe, test, expect } from 'vitest';
import {
  applyPostTransform,
  applyActivation,
  matmulBias,
  tensorToData,
  tensorRows,
  tensorCols,
} from '../../src/engine/ops.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function f64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

/** Check that every element of `actual` is within `tol` of `expected`. */
function expectClose(actual: Float64Array, expected: number[], tol = 1e-10): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], -Math.log10(tol));
  }
}

// ── applyPostTransform ────────────────────────────────────────────────────────

describe('applyPostTransform', () => {
  test('IDENTITY / undefined — returns input unchanged', () => {
    const x = f64(1, 2, 3);
    expect(applyPostTransform(x, 'IDENTITY', 3)).toBe(x);
    expect(applyPostTransform(x, undefined, 3)).toBe(x);
    expect(applyPostTransform(x, 'POST_TRANSFORM_UNSPECIFIED', 3)).toBe(x);
  });

  test('SIGMOID maps 0 → 0.5, large positive → ~1, large negative → ~0', () => {
    const out = applyPostTransform(f64(0, 100, -100), 'SIGMOID', 1);
    expectClose(out, [0.5, 1.0, 0.0], 1e-6);
  });

  test('SOFTMAX rows sum to 1 and preserve argmax', () => {
    // 2 rows × 3 classes
    const logits = f64(1, 2, 3,   3, 1, 2);
    const out = applyPostTransform(logits, 'SOFTMAX', 3);
    expect(out.length).toBe(6);
    const sum0 = out[0] + out[1] + out[2];
    const sum1 = out[3] + out[4] + out[5];
    expect(sum0).toBeCloseTo(1, 10);
    expect(sum1).toBeCloseTo(1, 10);
    // argmax of row 0 is index 2, row 1 is index 0
    expect(out[2]).toBeGreaterThan(out[1]);
    expect(out[3]).toBeGreaterThan(out[4]);
  });

  test('SOFTMAX is numerically stable with large values', () => {
    const out = applyPostTransform(f64(1000, 1001, 1002), 'SOFTMAX', 3);
    const sum = out[0] + out[1] + out[2];
    expect(isFinite(sum)).toBe(true);
    expect(sum).toBeCloseTo(1, 10);
  });

  test('SIGMOID_BINARY expands 1-column scores to [1-p, p] pairs', () => {
    const out = applyPostTransform(f64(0, 0), 'SIGMOID_BINARY', 1);
    // 2 rows, each score = 0 → p = 0.5
    expect(out.length).toBe(4);
    expectClose(out, [0.5, 0.5, 0.5, 0.5], 1e-10);
  });

  test('EXP applies elementwise exp', () => {
    const out = applyPostTransform(f64(0, 1, -1), 'EXP', 1);
    expectClose(out, [1, Math.E, 1 / Math.E], 1e-10);
  });

  test('LOGIT is inverse of sigmoid', () => {
    // logit(sigmoid(x)) ≈ x
    const x = f64(0, 1, -1, 2, -2);
    const sigX = applyPostTransform(x, 'SIGMOID', 1);
    const back = applyPostTransform(sigX, 'LOGIT', 1);
    expectClose(back, Array.from(x), 1e-8);
  });

  test('CLOGLOG maps 0 → ~0.632 (1 - 1/e)', () => {
    const out = applyPostTransform(f64(0), 'CLOGLOG', 1);
    expectClose(out, [1 - 1 / Math.E], 1e-10);
  });

  test('LOGLOG maps 0 → ~0.368 (1/e)', () => {
    const out = applyPostTransform(f64(0), 'LOGLOG', 1);
    expectClose(out, [1 / Math.E], 1e-10);
  });

  test('CAUCHIT maps 0 → 0.5', () => {
    const out = applyPostTransform(f64(0), 'CAUCHIT', 1);
    expectClose(out, [0.5], 1e-10);
  });
});

// ── applyActivation ───────────────────────────────────────────────────────────

describe('applyActivation', () => {
  test('IDENTITY returns input unchanged', () => {
    const x = f64(1, 2, 3);
    expect(applyActivation(x, 'IDENTITY', 3)).toBe(x);
  });

  test('LOGISTIC is element-wise sigmoid', () => {
    const out = applyActivation(f64(0, 100, -100), 'LOGISTIC', 1);
    expectClose(out, [0.5, 1.0, 0.0], 1e-6);
  });

  test('TANH maps 0 → 0', () => {
    const out = applyActivation(f64(0, 1, -1), 'TANH', 1);
    expectClose(out, [0, Math.tanh(1), Math.tanh(-1)], 1e-10);
  });

  test('RELU clips negatives', () => {
    const out = applyActivation(f64(-2, -1, 0, 1, 2), 'RELU', 1);
    expectClose(out, [0, 0, 0, 1, 2], 1e-10);
  });

  test('SOFTMAX rows sum to 1', () => {
    const out = applyActivation(f64(1, 2, 3), 'SOFTMAX', 3);
    expect(out[0] + out[1] + out[2]).toBeCloseTo(1, 10);
  });

  test('unknown activation returns input unchanged', () => {
    const x = f64(1, 2);
    expect(applyActivation(x, 'NOT_REAL', 2)).toBe(x);
  });
});

// ── matmulBias ────────────────────────────────────────────────────────────────

describe('matmulBias', () => {
  test('1×2 input @ 3×2 weight + 3 bias = 1×3 output', () => {
    // input: [[1, 2]]   weight (outD×inD): [[1,0],[0,1],[1,1]]  bias: [10,20,30]
    const input   = f64(1, 2);
    const weights = f64(1, 0,  0, 1,  1, 1);  // 3 rows × 2 cols
    const bias    = f64(10, 20, 30);
    const out = matmulBias(input, 1, 2, weights, 3, bias);
    // row·col: [1*1+2*0, 1*0+2*1, 1*1+2*1] + bias = [1,2,3] + [10,20,30]
    expectClose(out, [11, 22, 33], 1e-10);
  });

  test('2 rows × 3 outputs, no bias', () => {
    // 2×2 input, 3×2 weight → 2×3 output
    const input   = f64(1, 0,   0, 1);      // 2 rows, inD=2
    const weights = f64(1, 2,  3, 4,  5, 6); // 3 rows (outD), inD=2
    const out = matmulBias(input, 2, 2, weights, 3, null);
    // row0: [1*1+0*2, 1*3+0*4, 1*5+0*6] = [1,3,5]
    // row1: [0*1+1*2, 0*3+1*4, 0*5+1*6] = [2,4,6]
    expectClose(out, [1, 3, 5, 2, 4, 6], 1e-10);
  });

  test('zero weights → all-zero output', () => {
    // inD=3, outD=2 → weight matrix is 2×3 = 6 zeros
    const out = matmulBias(f64(1, 2, 3), 1, 3, f64(0, 0, 0, 0, 0, 0), 2, null);
    expectClose(out, [0, 0], 1e-10);
  });
});

// ── tensorToData ──────────────────────────────────────────────────────────────

describe('tensorToData', () => {
  test('plain number array → FLOAT64 1-D tensor', () => {
    const td = tensorToData([1, 2, 3]);
    expect(td.dtype).toBe('FLOAT64');
    expect(td.shape).toEqual([3]);
    expect(Array.from(td.data as Float64Array)).toEqual([1, 2, 3]);
  });

  test('Tensor with float32_data', () => {
    const td = tensorToData({
      type: { dtype: 'FLOAT32', shape: [2, 2] },
      float32_data: [1, 2, 3, 4],
    });
    expect(td.dtype).toBe('FLOAT32');
    expect(td.shape).toEqual([2, 2]);
    expect(Array.from(td.data as Float64Array)).toEqual([1, 2, 3, 4]);
  });

  test('Tensor with string_data', () => {
    const td = tensorToData({
      type: { dtype: 'STRING', shape: [3] },
      string_data: ['a', 'b', 'c'],
    });
    expect(td.dtype).toBe('STRING');
    expect(td.data).toEqual(['a', 'b', 'c']);
  });

  test('tensorRows and tensorCols', () => {
    const td = tensorToData({ type: { dtype: 'FLOAT64', shape: [4, 3] }, float64_data: new Array(12).fill(0) });
    expect(tensorRows(td)).toBe(4);
    expect(tensorCols(td)).toBe(3);
  });
});
