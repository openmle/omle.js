// Unit tests for engine/nn.ts — feed-forward neural network (MLP) execution.

import { describe, test, expect } from 'vitest';
import { executeNeuralNetwork } from '../../../src/engine/nn.js';
import type { NeuralNetwork } from '../../../src/ir.js';
import type { ResolvedModel } from '../../../src/resolve.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function emptyResolved(): ResolvedModel {
  return { model: { inputs: [] }, tensorIndex: new Map(), executionOrder: [], schemaNames: new Set() };
}

function f64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

/** Inline TensorValue for layer weights/biases. */
function tv(data: number[], shape: number[]) {
  return { tensor: { type: { dtype: 'FLOAT64', shape }, float64_data: data } };
}

function expectClose(actual: Float64Array, expected: number[], tol = 1e-7): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], -Math.log10(tol));
  }
}

// ── Single layer ──────────────────────────────────────────────────────────────

describe('executeNeuralNetwork — single layer', () => {
  test('identity activation: output = W @ x + b', () => {
    // 1 sample, 2 inputs → 1 output
    // W = [[2, 3]], b = [1]  →  output = 2*1 + 3*2 + 1 = 9
    const nn: NeuralNetwork = {
      layers: [{
        weights: tv([2, 3], [1, 2]),
        bias:    tv([1], [1]),
        activation: 'IDENTITY',
      }],
    };
    const out = executeNeuralNetwork(nn, f64(1, 2), 2, 1, emptyResolved());
    expectClose(out, [9]);
  });

  test('RELU activation clips negatives', () => {
    // W = [[-1, 0], [0, 1]], b = [5, -5]
    // x = [1, 1]: raw = [-1+5, 1-5] = [4, -4]; relu → [4, 0]
    const nn: NeuralNetwork = {
      layers: [{
        weights: tv([-1, 0,  0, 1], [2, 2]),
        bias:    tv([5, -5], [2]),
        activation: 'RELU',
      }],
    };
    const out = executeNeuralNetwork(nn, f64(1, 1), 2, 1, emptyResolved());
    expectClose(out, [4, 0]);
  });

  test('LOGISTIC activation maps 0 → 0.5', () => {
    // W = [[0, 0]], no bias → raw = 0 → logistic = 0.5
    const nn: NeuralNetwork = {
      layers: [{
        weights: tv([0, 0], [1, 2]),
        activation: 'LOGISTIC',
      }],
    };
    const out = executeNeuralNetwork(nn, f64(1, 2), 2, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(0.5);
  });

  test('TANH activation maps 0 → 0', () => {
    const nn: NeuralNetwork = {
      layers: [{
        weights: tv([0, 0], [1, 2]),
        activation: 'TANH',
      }],
    };
    const out = executeNeuralNetwork(nn, f64(1, 2), 2, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(0);
  });

  test('no bias: output = W @ x', () => {
    const nn: NeuralNetwork = {
      layers: [{
        weights: tv([1, 2], [1, 2]),  // W = [[1, 2]]
        // no bias field
      }],
    };
    const out = executeNeuralNetwork(nn, f64(3, 4), 2, 1, emptyResolved());
    // 1*3 + 2*4 = 11
    expectClose(out, [11]);
  });
});

// ── Two layers ────────────────────────────────────────────────────────────────

describe('executeNeuralNetwork — two layers', () => {
  test('2→2→1 MLP: output chained through layers', () => {
    // Layer 0: W=[[1,0],[0,1]], b=[0,0], RELU: output = relu(x) = x (for positive x)
    // Layer 1: W=[[1,1]], b=[0], IDENTITY: output = sum of layer-0 outputs
    const nn: NeuralNetwork = {
      layers: [
        {
          weights:    tv([1, 0,  0, 1], [2, 2]),
          bias:       tv([0, 0], [2]),
          activation: 'RELU',
        },
        {
          weights:    tv([1, 1], [1, 2]),
          activation: 'IDENTITY',
        },
      ],
    };
    // x = [3, 4]: layer0 → [3, 4]; layer1 → 3+4 = 7
    const out = executeNeuralNetwork(nn, f64(3, 4), 2, 1, emptyResolved());
    expectClose(out, [7]);
  });

  test('SOFTMAX final layer: outputs sum to 1', () => {
    // 1→2 layer with SOFTMAX so probabilities sum to 1
    const nn: NeuralNetwork = {
      layers: [
        {
          weights:    tv([1, -1], [2, 1]),  // 2 outputs from 1 input
          activation: 'SOFTMAX',
        },
      ],
    };
    const out = executeNeuralNetwork(nn, f64(1), 1, 1, emptyResolved());
    expect(out[0] + out[1]).toBeCloseTo(1, 10);
  });
});

// ── Batch ─────────────────────────────────────────────────────────────────────

describe('executeNeuralNetwork — batch', () => {
  test('2 samples processed independently', () => {
    // W = [[1, 0]], b = [0]: output = first feature
    const nn: NeuralNetwork = {
      layers: [{
        weights: tv([1, 0], [1, 2]),
        activation: 'IDENTITY',
      }],
    };
    // row0 = [5, 9] → 5; row1 = [7, 3] → 7
    const out = executeNeuralNetwork(nn, f64(5, 9,  7, 3), 2, 2, emptyResolved());
    expectClose(out, [5, 7]);
  });
});

// ── Edge cases ────────────────────────────────────────────────────────────────

describe('executeNeuralNetwork — edge cases', () => {
  test('no layers returns input unchanged', () => {
    const nn: NeuralNetwork = { layers: [] };
    const input = f64(1, 2, 3);
    const out = executeNeuralNetwork(nn, input, 3, 1, emptyResolved());
    expect(out).toBe(input);  // same reference — no copy
  });
});
