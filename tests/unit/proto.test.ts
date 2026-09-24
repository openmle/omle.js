// Unit tests for src/proto.ts — decoding the binary .omle wire format.
//
// Fixtures are encoded here with the same embedded descriptor, so the tests
// stay self-contained: no .omle file or sibling checkout is needed.

import { describe, test, expect } from 'vitest';
import protobufLight from 'protobufjs/light.js';
import { fromProtoBinary } from '../../src/proto.js';
import { protoDescriptor } from '../../src/proto_descriptor.js';

const root = protobufLight.Root.fromJSON(protoDescriptor);
const ModelType = root.lookupType('omle.OMLEModel');

function encode(model: object): Uint8Array {
  return ModelType.encode(ModelType.fromObject(model)).finish();
}

describe('fromProtoBinary', () => {
  test('decodes a model round-tripped through the wire format', () => {
    const decoded = fromProtoBinary(encode({
      metadata: { name: 'round-trip' },
      inputs: [{ name: 'x', type: { dtype: 'FLOAT32', shape: [4] } }],
    }));
    expect(decoded.metadata?.name).toBe('round-trip');
    expect(decoded.inputs?.[0]?.name).toBe('x');
    expect(decoded.inputs?.[0]?.type?.shape).toEqual([4]);
  });

  test('enums decode as strings, matching the JSON convention', () => {
    const decoded = fromProtoBinary(encode({
      inputs: [{ name: 'x', type: { dtype: 'FLOAT64', shape: [] } }],
    }));
    expect(decoded.inputs?.[0]?.type?.dtype).toBe('FLOAT64');
  });

  test('typed list-wrappers are flattened to plain arrays', () => {
    // On the wire float32_data is a Float32List message, not a bare array.
    const decoded = fromProtoBinary(encode({
      tensor_entries: [{
        id: 't1',
        dense: {
          type: { dtype: 'FLOAT32', shape: [3] },
          float32_data: { values: [1, 2, 3] },
        },
      }],
    }));
    expect(decoded.tensor_entries?.[0]?.dense?.float32_data).toEqual([1, 2, 3]);
  });

  test('DiscreteDomain.values stays structured, not collapsed', () => {
    const decoded = fromProtoBinary(encode({
      model_schema: {
        features: [{
          name: 'f',
          // DomainValue wraps a Scalar in its `value` field.
          domain: {
            discrete: {
              values: [{ value: { string_value: 'a' } }, { value: { string_value: 'b' } }],
            },
          },
        }],
      },
    }));
    const values = decoded.model_schema?.features?.[0]?.domain?.discrete?.values;
    expect(values).toHaveLength(2);
    expect(values?.[0]?.value?.string_value).toBe('a');
  });

  test('accepts both Uint8Array and ArrayBuffer', () => {
    const bytes = encode({ metadata: { name: 'buf' } });
    const copy = new Uint8Array(bytes);           // own the buffer exactly
    expect(fromProtoBinary(copy).metadata?.name).toBe('buf');
    expect(fromProtoBinary(copy.buffer).metadata?.name).toBe('buf');
  });

  test('an empty message decodes to an empty model', () => {
    expect(fromProtoBinary(encode({}))).toEqual({});
  });
});
