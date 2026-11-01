// Unit tests for engine/validate_inputs.ts

import { describe, test, expect } from 'vitest';
import { validateInputs } from '../../../src/engine/validate_inputs.js';
import type { OMLEModel } from '../../../src/ir.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function model(inputs: OMLEModel['inputs']): OMLEModel {
  return { inputs };
}

function modelWithFeatures(features: OMLEModel['model_schema']): OMLEModel {
  return { model_schema: features };
}

// ── No declared inputs → always valid ────────────────────────────────────────

describe('validateInputs — no declared inputs', () => {
  test('empty inputs array: always valid', () => {
    const result = validateInputs(model([]), { anything: 42 });
    expect(result.valid).toBe(true);
    expect(result.issues).toHaveLength(0);
  });

  test('no inputs field at all: always valid', () => {
    const result = validateInputs({}, {});
    expect(result.valid).toBe(true);
  });
});

// ── Presence check ────────────────────────────────────────────────────────────

describe('validateInputs — presence', () => {
  const m = model([{ name: 'age' }, { name: 'income' }]);

  test('all inputs present: valid', () => {
    const result = validateInputs(m, { age: 25, income: 50000 });
    expect(result.valid).toBe(true);
    expect(result.issues).toHaveLength(0);
  });

  test('missing one input: invalid, one issue', () => {
    const result = validateInputs(m, { age: 25 });
    expect(result.valid).toBe(false);
    expect(result.issues).toHaveLength(1);
    expect(result.issues[0].kind).toBe('missing');
    expect(result.issues[0].name).toBe('income');
  });

  test('missing all inputs: one issue per missing', () => {
    const result = validateInputs(m, {});
    expect(result.valid).toBe(false);
    expect(result.issues).toHaveLength(2);
    expect(result.issues.every(i => i.kind === 'missing')).toBe(true);
  });

  test('null value treated as missing', () => {
    const result = validateInputs(m, { age: null, income: 50000 });
    expect(result.valid).toBe(false);
    expect(result.issues[0].kind).toBe('missing');
    expect(result.issues[0].name).toBe('age');
  });
});

// ── Dtype checks ──────────────────────────────────────────────────────────────

describe('validateInputs — dtype', () => {
  test('numeric value for FLOAT64: valid', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [] } }]);
    expect(validateInputs(m, { x: 3.14 }).valid).toBe(true);
  });

  test('string value for STRING: valid', () => {
    const m = model([{ name: 'label', type: { dtype: 'STRING', shape: [] } }]);
    expect(validateInputs(m, { label: 'cat' }).valid).toBe(true);
  });

  test('string value for FLOAT64: wrong_type issue', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [] } }]);
    const result = validateInputs(m, { x: 'oops' });
    expect(result.valid).toBe(false);
    expect(result.issues[0].kind).toBe('wrong_type');
    expect(result.issues[0].name).toBe('x');
  });

  test('Float64Array for FLOAT64: valid', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [-1] } }]);
    expect(validateInputs(m, { x: new Float64Array([1, 2, 3]) }).valid).toBe(true);
  });

  test('number array for STRING: wrong_type', () => {
    const m = model([{ name: 'txt', type: { dtype: 'STRING', shape: [] } }]);
    const result = validateInputs(m, { txt: 42 });
    expect(result.valid).toBe(false);
    expect(result.issues[0].kind).toBe('wrong_type');
  });
});

// ── Shape checks ──────────────────────────────────────────────────────────────

describe('validateInputs — shape', () => {
  test('exact shape match: valid', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [3] } }]);
    expect(validateInputs(m, { x: [1, 2, 3] }).valid).toBe(true);
  });

  test('wrong fixed shape: wrong_shape issue', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [3] } }]);
    const result = validateInputs(m, { x: [1, 2] });
    expect(result.valid).toBe(false);
    expect(result.issues[0].kind).toBe('wrong_shape');
    expect(result.issues[0].name).toBe('x');
  });

  test('dynamic first dim (-1): any length accepted', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [-1] } }]);
    expect(validateInputs(m, { x: [1, 2, 3, 4, 5] }).valid).toBe(true);
    expect(validateInputs(m, { x: [1] }).valid).toBe(true);
  });

  test('dynamic batch dim [-1, 3]: flat array of 3 accepted as single row', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [-1, 3] } }]);
    expect(validateInputs(m, { x: [1, 2, 3] }).valid).toBe(true);
  });

  test('scalar compatible with dynamic-only shape: valid', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [-1] } }]);
    expect(validateInputs(m, { x: 5 }).valid).toBe(true);
  });

  test('scalar incompatible with fixed-size shape: wrong_shape', () => {
    const m = model([{ name: 'x', type: { dtype: 'FLOAT64', shape: [3] } }]);
    const result = validateInputs(m, { x: 5 });
    expect(result.valid).toBe(false);
    expect(result.issues[0].kind).toBe('wrong_shape');
  });
});

// ── model_schema.features fallback ───────────────────────────────────────────

describe('validateInputs — model_schema.features fallback', () => {
  test('validates named features when model.inputs is absent', () => {
    const m = modelWithFeatures({
      features: [
        { name: 'age',    type: { dtype: 'FLOAT64', shape: [] } },
        { name: 'income', type: { dtype: 'FLOAT64', shape: [] } },
      ],
    });
    expect(validateInputs(m, { age: 30, income: 50000 }).valid).toBe(true);
    const result = validateInputs(m, { age: 30 });
    expect(result.valid).toBe(false);
    expect(result.issues[0].name).toBe('income');
  });
});
