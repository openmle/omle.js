// Unit tests for engine/svm.ts — linear SVM and RBF kernel SVM execution.

import { describe, test, expect } from 'vitest';
import { executeSVM } from '../../../src/engine/svm.js';
import type { SVM } from '../../../src/ir.js';
import type { ResolvedModel } from '../../../src/resolve.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function emptyResolved(): ResolvedModel {
  return { model: { inputs: [] }, tensorIndex: new Map(), executionOrder: [], schemaNames: new Set() };
}

function f64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

function tv(data: number[], shape: number[]) {
  return { tensor: { type: { dtype: 'FLOAT64', shape }, float64_data: data } };
}

function expectClose(actual: Float64Array, expected: number[], tol = 1e-7): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], -Math.log10(tol));
  }
}

// ── Linear SVM ───────────────────────────────────────────────────────────────

describe('executeSVM — linear', () => {
  test('binary: score = X @ w + b', () => {
    // coeff = [1, 2], intercept = [-3]  → score = 1*2 + 2*3 - 3 = 5
    const svm: SVM = {
      linear: {
        coefficients: tv([1, 2], [1, 2]),
        intercept:    tv([-3], [1]),
      },
    };
    const out = executeSVM(svm, f64(2, 3), 2, 1, emptyResolved());
    expectClose(out, [5]);
  });

  test('batch: 2 samples', () => {
    // coeff = [1, 0]: output = first feature
    const svm: SVM = {
      linear: {
        coefficients: tv([1, 0], [1, 2]),
        intercept:    tv([0], [1]),
      },
    };
    // row0=[5, 9]→5; row1=[3, 7]→3
    const out = executeSVM(svm, f64(5, 9,  3, 7), 2, 2, emptyResolved());
    expectClose(out, [5, 3]);
  });

  test('multiclass: coeff shape [numClasses, numFeatures]', () => {
    // 3 classes, 2 features; coeff = [[1,0],[0,1],[1,1]], intercept = [0,0,0]
    const svm: SVM = {
      linear: {
        coefficients: tv([1, 0,  0, 1,  1, 1], [3, 2]),
        intercept:    tv([0, 0, 0], [3]),
      },
    };
    // x = [3, 4] → scores = [3, 4, 7]
    const out = executeSVM(svm, f64(3, 4), 2, 1, emptyResolved());
    expectClose(out, [3, 4, 7]);
  });

  test('SIGMOID post_transform applied', () => {
    // coeff = [[0]], intercept = [0] → raw score = 0 → sigmoid = 0.5
    const svm: SVM = {
      linear: {
        coefficients: tv([0], [1, 1]),
        intercept:    tv([0], [1]),
      },
      post_transform: 'SIGMOID',
    };
    const out = executeSVM(svm, f64(1), 1, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(0.5);
  });
});

// ── Kernel SVM (RBF) ──────────────────────────────────────────────────────────

describe('executeSVM — RBF kernel', () => {
  // One support vector at [0, 0], dual coeff = [1], bias = 0, gamma = 1.
  // K(x, sv) = exp(-gamma * ||x - sv||^2) = exp(-||x||^2)
  // decision(x) = dual * K + bias = exp(-||x||^2)

  function rbfSVM(gamma: number, svData: number[], nSV: number, nFeat: number, dualData: number[]): SVM {
    return {
      kernel: {
        kernel_type:      'RBF',
        gamma:            { double_value: gamma },
        support_vectors:  tv(svData, [nSV, nFeat]),
        dual_coefficients: tv(dualData, [1, nSV]),
        intercept:         tv([0], [1]),
      },
    };
  }

  test('single SV at origin: decision = exp(-||x||^2)', () => {
    const svm = rbfSVM(1.0, [0, 0], 1, 2, [1.0]);
    // x = [1, 0]: K = exp(-1) ≈ 0.368
    const out = executeSVM(svm, f64(1, 0), 2, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(Math.exp(-1), 6);
  });

  test('x = SV: K = 1 (maximum similarity)', () => {
    const svm = rbfSVM(1.0, [3, 4], 1, 2, [1.0]);
    // x = [3, 4] = SV → ||x - sv||^2 = 0 → K = exp(0) = 1
    const out = executeSVM(svm, f64(3, 4), 2, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(1.0, 6);
  });

  test('gamma=0 (edge): K = exp(0) = 1 regardless of distance', () => {
    const svm = rbfSVM(0.0, [0, 0], 1, 2, [2.0]);
    // K = exp(0) = 1; decision = 2 * 1 = 2
    const out = executeSVM(svm, f64(10, 10), 2, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(2.0, 6);
  });

  test('batch: two samples, independent kernel evaluations', () => {
    const svm = rbfSVM(1.0, [0, 0], 1, 2, [1.0]);
    // row0=[0,0]: K=1; row1=[1,0]: K=exp(-1)
    const out = executeSVM(svm, f64(0, 0,  1, 0), 2, 2, emptyResolved());
    expect(out[0]).toBeCloseTo(1.0, 6);
    expect(out[1]).toBeCloseTo(Math.exp(-1), 6);
  });
});

// ── Kernel SVM (LINEAR kernel) ────────────────────────────────────────────────

describe('executeSVM — LINEAR kernel', () => {
  test('K(x, sv) = x · sv; decision = dual * K + bias', () => {
    // SV = [1, 2], dual = [3], bias = 1
    // x = [2, 3]: K = 1*2 + 2*3 = 8; decision = 3*8 + 1 = 25
    const svm: SVM = {
      kernel: {
        kernel_type:       'LINEAR',
        support_vectors:   tv([1, 2], [1, 2]),
        dual_coefficients: tv([3], [1, 1]),
        intercept:         tv([1], [1]),
      },
    };
    const out = executeSVM(svm, f64(2, 3), 2, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(25, 6);
  });
});

// ── Empty SVM ─────────────────────────────────────────────────────────────────

describe('executeSVM — fallthrough', () => {
  test('SVM with no linear/kernel returns zeros', () => {
    const svm: SVM = {};
    const out = executeSVM(svm, f64(1, 2), 2, 1, emptyResolved());
    expect(out.length).toBe(1);
    expect(out[0]).toBe(0);
  });
});
