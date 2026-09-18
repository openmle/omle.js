// Unit tests for model-specific execute functions.
// Each function is exercised with hand-crafted minimal inputs so failures
// point at the specific model logic rather than the full engine pipeline.

import { describe, test, expect } from 'vitest';
import { executeLinear } from '../../../src/engine/linear.js';
import { executeNaiveBayes } from '../../../src/engine/naive_bayes.js';
import type { Linear, NaiveBayes } from '../../../src/ir.js';
import type { ResolvedModel } from '../../../src/resolve.js';

// ── Test helpers ──────────────────────────────────────────────────────────────

/** Build a minimal ResolvedModel with an empty tensor index. */
function emptyResolved(): ResolvedModel {
  return {
    model: { inputs: [] },
    tensorIndex: new Map(),
    executionOrder: [],
    schemaNames: new Set(),
  };
}

/** Wrap a flat number array as an inline TensorValue. */
function tv(data: number[], shape: number[]): import('../../../src/ir.js').TensorValue {
  return { tensor: { type: { dtype: 'FLOAT64', shape }, float64_data: data } };
}

function f64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

function expectClose(actual: Float64Array, expected: number[], tol = 1e-8): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    if (!isFinite(expected[i])) {
      expect(isFinite(actual[i])).toBe(false);
    } else {
      expect(actual[i]).toBeCloseTo(expected[i], -Math.log10(tol));
    }
  }
}

// ── executeLinear ─────────────────────────────────────────────────────────────

describe('executeLinear', () => {
  test('single output: score = X @ w + b', () => {
    // 2 samples, 2 features; coeff = [3, 1]; intercept = [0.5]
    // scores = [1*3 + 2*1, 3*3 + 4*1] + 0.5 = [5.5, 13.5]
    const linear: Linear = {
      coefficients: tv([3, 1], [2]),
      intercept:    tv([0.5], [1]),
    };
    const inputs = f64(1, 2,  3, 4);
    const out = executeLinear(linear, inputs, 2, 2, emptyResolved());
    expectClose(out, [5.5, 13.5]);
  });

  test('multiclass: score = X @ W^T + b, shape [outW, inD]', () => {
    // 1 sample, 2 features; W = [[1,0],[0,1],[1,1]], b = [0,0,0]
    // score = [1, 2, 3]
    const linear: Linear = {
      coefficients: tv([1, 0,  0, 1,  1, 1], [3, 2]),
    };
    const out = executeLinear(linear, f64(1, 2), 2, 1, emptyResolved());
    expectClose(out, [1, 2, 3]);
  });

  test('SIGMOID post_transform maps score to (0,1)', () => {
    // score = 0 → sigmoid = 0.5
    const linear: Linear = {
      coefficients: tv([0], [1]),
      post_transform: 'SIGMOID',
    };
    const out = executeLinear(linear, f64(1), 1, 1, emptyResolved());
    expectClose(out, [0.5]);
  });

  test('SOFTMAX post_transform: rows sum to 1', () => {
    const linear: Linear = {
      coefficients: tv([1, 0, 0,  0, 1, 0,  0, 0, 1], [3, 3]),
      post_transform: 'SOFTMAX',
    };
    // identity weights → logits = input
    const out = executeLinear(linear, f64(1, 2, 3), 3, 1, emptyResolved());
    expect(out[0] + out[1] + out[2]).toBeCloseTo(1, 10);
    // argmax should be class 2 (highest logit)
    expect(out[2]).toBeGreaterThan(out[1]);
  });

  test('no intercept — bias is null', () => {
    const linear: Linear = { coefficients: tv([2], [1]) };
    const out = executeLinear(linear, f64(3), 1, 1, emptyResolved());
    expectClose(out, [6]);
  });
});

// ── executeNaiveBayes ─────────────────────────────────────────────────────────

describe('executeNaiveBayes — Gaussian', () => {
  test('single feature, 2 classes: predict the closer class', () => {
    // Class 0: mean=0, var=1; Class 1: mean=10, var=1
    // x=0 → class 0 wins (higher posterior); x=10 → class 1 wins
    const nb: NaiveBayes = {
      class_log_priors: tv([Math.log(0.5), Math.log(0.5)], [2]),
      gaussian: {
        means:     tv([0, 10], [2, 1]),  // [numClasses, numFeatures]
        variances: tv([1,  1], [2, 1]),
      },
    };

    const out0 = executeNaiveBayes(nb, f64(0), 1, 1, emptyResolved());
    expect(out0[0]).toBeGreaterThan(out0[1]);  // class 0 probability > class 1

    const out10 = executeNaiveBayes(nb, f64(10), 1, 1, emptyResolved());
    expect(out10[1]).toBeGreaterThan(out10[0]);  // class 1 probability > class 0
  });

  test('output probabilities sum to 1 (SOFTMAX applied)', () => {
    const nb: NaiveBayes = {
      class_log_priors: tv([Math.log(0.5), Math.log(0.5)], [2]),
      gaussian: {
        means:     tv([0, 5], [2, 1]),
        variances: tv([1, 1], [2, 1]),
      },
    };
    const out = executeNaiveBayes(nb, f64(2), 1, 1, emptyResolved());
    expect(out[0] + out[1]).toBeCloseTo(1, 10);
  });

  test('batch: 2 rows produce independent predictions', () => {
    const nb: NaiveBayes = {
      class_log_priors: tv([Math.log(0.5), Math.log(0.5)], [2]),
      gaussian: {
        // [numClasses, numFeatures] row-major: class0=[0,0], class1=[10,10]
        means:     tv([0, 0, 10, 10], [2, 2]),
        variances: tv([1, 1, 1, 1],   [2, 2]),
      },
    };
    // row0=[0,0] → class 0 wins;  row1=[10,10] → class 1 wins
    const out = executeNaiveBayes(nb, f64(0, 0,  10, 10), 2, 2, emptyResolved());
    expect(out[0]).toBeGreaterThan(out[1]);  // row 0: class 0
    expect(out[3]).toBeGreaterThan(out[2]);  // row 1: class 1
  });
});

describe('executeNaiveBayes — Multinomial', () => {
  test('class with higher log-prob for observed feature wins', () => {
    // feature_log_prob[class, feature]: 2 classes × 1 feature
    // class 0: log(0.9)≈-0.105;  class 1: log(0.1)≈-2.3
    const nb: NaiveBayes = {
      class_log_priors: tv([Math.log(0.5), Math.log(0.5)], [2]),
      multinomial: {
        feature_log_prob: tv([Math.log(0.9), Math.log(0.1)], [2, 1]),
      },
    };
    // feature count = 1 → class 0 should win
    const out = executeNaiveBayes(nb, f64(1), 1, 1, emptyResolved());
    expect(out[0]).toBeGreaterThan(out[1]);
  });
});

