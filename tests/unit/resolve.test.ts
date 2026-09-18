// Unit tests for src/resolve.ts — graph topology and name-expansion helpers.

import { describe, test, expect } from 'vitest';
import {
  expandNameRange,
  expandNodeInputs,
  expandFeatures,
  resolveRef,
  resolveRefDense,
  resolve,
} from '../../src/resolve.js';
import type { OMLEModel, TensorEntry, NameRange, NodeInput, Feature } from '../../src/ir.js';

// ── expandNameRange ───────────────────────────────────────────────────────────

describe('expandNameRange', () => {
  test('generates names from prefix + sequential index', () => {
    const r: NameRange = { prefix: 'x_', start: 0, end: 3 };
    expect(expandNameRange(r)).toEqual(['x_0', 'x_1', 'x_2']);
  });

  test('respects width padding', () => {
    const r: NameRange = { prefix: 'f', start: 0, end: 3, width: 3 };
    expect(expandNameRange(r)).toEqual(['f000', 'f001', 'f002']);
  });

  test('empty range returns []', () => {
    expect(expandNameRange({ prefix: 'x', start: 5, end: 5 })).toEqual([]);
  });

  test('non-zero start', () => {
    const r: NameRange = { prefix: 'col_', start: 2, end: 5 };
    expect(expandNameRange(r)).toEqual(['col_2', 'col_3', 'col_4']);
  });
});

// ── expandNodeInputs ──────────────────────────────────────────────────────────

describe('expandNodeInputs', () => {
  test('named inputs expand to their value strings', () => {
    const inputs: NodeInput[] = [
      { name: { value: 'a' } },
      { name: { value: 'b' } },
    ];
    expect(expandNodeInputs(inputs)).toEqual(['a', 'b']);
  });

  test('range inputs expand to all generated names', () => {
    const inputs: NodeInput[] = [
      { range: { prefix: 'f_', start: 0, end: 3 } },
    ];
    expect(expandNodeInputs(inputs)).toEqual(['f_0', 'f_1', 'f_2']);
  });

  test('mixed named and range', () => {
    const inputs: NodeInput[] = [
      { name: { value: 'bias' } },
      { range: { prefix: 'x_', start: 0, end: 2 } },
    ];
    expect(expandNodeInputs(inputs)).toEqual(['bias', 'x_0', 'x_1']);
  });

  test('empty list returns []', () => {
    expect(expandNodeInputs([])).toEqual([]);
  });
});

// ── expandFeatures ────────────────────────────────────────────────────────────

describe('expandFeatures', () => {
  test('named features pass through unchanged', () => {
    const feats: Feature[] = [
      { name: 'age' },
      { name: 'income' },
    ];
    expect(expandFeatures(feats)).toEqual(feats);
  });

  test('range features are expanded with correct names', () => {
    const feats: Feature[] = [
      { range: { prefix: 'emb_', start: 0, end: 3 } },
    ];
    const out = expandFeatures(feats);
    expect(out.map(f => f.name)).toEqual(['emb_0', 'emb_1', 'emb_2']);
  });

  test('range feature inherits non-name/range properties', () => {
    const feats: Feature[] = [
      { range: { prefix: 'x_', start: 0, end: 2 }, missing_value_policy: 'MISSING_AS_VALUE' },
    ];
    const out = expandFeatures(feats);
    expect(out[0].missing_value_policy).toBe('MISSING_AS_VALUE');
    expect(out[1].missing_value_policy).toBe('MISSING_AS_VALUE');
  });

  test('range feature sets index when base index is defined', () => {
    const feats: Feature[] = [
      { range: { prefix: 'f_', start: 0, end: 3 }, index: 10 },
    ];
    const out = expandFeatures(feats);
    expect(out.map(f => f.index)).toEqual([10, 11, 12]);
  });
});

// ── resolveRef ────────────────────────────────────────────────────────────────

describe('resolveRef', () => {
  const entry: TensorEntry = {
    id: 'te1',
    dense: { type: { dtype: 'FLOAT32', shape: [2] }, float32_data: [1, 2] },
  };
  const index = new Map([['te1', entry]]);

  test('returns entry for known id', () => {
    expect(resolveRef({ id: 'te1' }, index)).toBe(entry);
  });

  test('throws for unknown id', () => {
    expect(() => resolveRef({ id: 'missing' }, index)).toThrow('TensorRef not found');
  });

  test('resolveRefDense returns the dense tensor', () => {
    expect(resolveRefDense({ id: 'te1' }, index)).toBe(entry.dense);
  });
});

// ── resolve (graph topology) ──────────────────────────────────────────────────

describe('resolve', () => {
  test('tensorIndex maps entry ids', () => {
    const model: OMLEModel = {
      inputs: [{ name: 'x' }],
      tensor_entries: [{ id: 'te1', dense: { type: { dtype: 'FLOAT32', shape: [1] }, float32_data: [0] } }],
    };
    const r = resolve(model);
    expect(r.tensorIndex.has('te1')).toBe(true);
  });

  test('schemaNames includes input names', () => {
    const model: OMLEModel = { inputs: [{ name: 'a' }, { name: 'b' }] };
    const r = resolve(model);
    expect(r.schemaNames.has('a')).toBe(true);
    expect(r.schemaNames.has('b')).toBe(true);
  });

  test('executionOrder respects data-flow dependencies', () => {
    // Node B depends on node A's output
    const model: OMLEModel = {
      inputs: [{ name: 'x' }],
      nodes: [
        {
          name: 'B',
          op: 'openmle.linear',
          inputs: [{ name: { value: 'a_out' } }],  // produced by A
          outputs: [{ name: 'b_out' }],
        },
        {
          name: 'A',
          op: 'openmle.linear',
          inputs: [{ name: { value: 'x' } }],
          outputs: [{ name: 'a_out' }],
        },
      ],
    };
    const r = resolve(model);
    const names = r.executionOrder.map(n => n.name);
    expect(names.indexOf('A')).toBeLessThan(names.indexOf('B'));
  });
});
