// Unit tests for src/io.ts — fromJSON / toJSON without any file I/O.

import { describe, test, expect } from 'vitest';
import { fromJSON, toJSON } from '../../src/io.js';
import type { OMLEModel } from '../../src/ir.js';

// ── fromJSON ──────────────────────────────────────────────────────────────────

describe('fromJSON', () => {
  test('accepts a plain object', () => {
    const model = fromJSON({ inputs: [{ name: 'x' }] });
    expect(model.inputs?.[0]?.name).toBe('x');
  });

  test('accepts a JSON string', () => {
    const model = fromJSON('{"inputs":[{"name":"y"}]}');
    expect(model.inputs?.[0]?.name).toBe('y');
  });

  test('returns empty object for {}', () => {
    const model = fromJSON({});
    expect(model).toEqual({});
  });

  // ── flattenListWrappers ────────────────────────────────────────────────────

  test('flattens scalar {values:[...]} list-wrappers', () => {
    // float32_data can arrive as { values: [1, 2, 3] } from the Python SDK
    const model = fromJSON({
      tensor_entries: [{
        id: 't1',
        dense: {
          type: { dtype: 'FLOAT32', shape: [3] },
          float32_data: { values: [1, 2, 3] },
        },
      }],
    });
    expect(model.tensor_entries?.[0]?.dense?.float32_data).toEqual([1, 2, 3]);
  });

  test('does NOT flatten object-element arrays (e.g. DiscreteDomain.values)', () => {
    // DomainValue messages must stay as objects, not be unwrapped
    const domainValues = [{ value: 'a' }, { value: 'b' }];
    const model = fromJSON({
      model_schema: {
        features: [{
          name: 'f',
          domain: {
            discrete: {
              values: domainValues,
            },
          },
        }],
      },
    });
    const disc = model.model_schema?.features?.[0]?.domain?.discrete;
    expect(disc?.values).toEqual(domainValues);
  });

  test('flattens nested list-wrappers recursively', () => {
    const model = fromJSON({
      nodes: [{
        name: 'n1',
        linear: {
          coefficients: { float64_data: { values: [0.1, 0.2] } },
        },
      }],
    });
    const coeff = model.nodes?.[0]?.linear?.coefficients;
    expect((coeff as Record<string, unknown>)?.float64_data).toEqual([0.1, 0.2]);
  });

  test('handles null and undefined gracefully', () => {
    const model = fromJSON({ inputs: null as unknown, outputs: undefined as unknown } as Record<string, unknown>);
    expect(model.inputs).toBeNull();
  });
});

// ── toJSON / round-trip ───────────────────────────────────────────────────────

describe('toJSON', () => {
  test('serialises and round-trips a minimal model', () => {
    const original: OMLEModel = {
      inputs: [{ name: 'x' }],
      outputs: [{ name: 'y' }],
    };
    const json = toJSON(original);
    const restored = fromJSON(json);
    expect(restored.inputs?.[0]?.name).toBe('x');
    expect(restored.outputs?.[0]?.name).toBe('y');
  });

  test('pretty flag produces indented output', () => {
    const model: OMLEModel = { inputs: [{ name: 'x' }] };
    const compact = toJSON(model, false);
    const pretty  = toJSON(model, true);
    expect(pretty).toContain('\n');
    expect(compact).not.toContain('\n');
  });

  test('omits null and undefined values', () => {
    const model = fromJSON({ inputs: [{ name: 'x', type: null }] } as Record<string, unknown>);
    const json = toJSON(model);
    // null / undefined fields should not appear in the output
    expect(json).not.toContain('"type":null');
  });
});
