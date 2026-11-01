// Binary .omle loader — decodes the protobuf wire format.
//
// Wire format is not self-describing: the bytes carry field numbers, not names
// or types, so decoding needs the schema. It is embedded as a protobufjs JSON
// descriptor (src/proto_descriptor.ts, generated from omle.proto) rather than
// the .proto text, which is ~4x larger and would also drag in the .proto
// grammar parser. protobufjs/light provides reflection without that parser.

import protobufLight from 'protobufjs/light.js';
import type { Root } from 'protobufjs/light.js';
import { protoDescriptor } from './proto_descriptor.js';
import { fromJSON } from './io.js';
import type { OMLEModel } from './ir.js';

let _root: Root | null = null;

function getRoot(): Root {
  if (!_root) {
    _root = protobufLight.Root.fromJSON(protoDescriptor);
  }
  return _root;
}

/**
 * Decode a binary `.omle` file into an OMLEModel.
 *
 * Returns a model identical to what `fromJSON` produces for the same model, so
 * callers can treat the two encodings interchangeably. The proto encoding wraps
 * repeated typed arrays in list messages (Float32List and friends); fromJSON
 * already unwraps those, so decoding hands its output straight to it.
 */
export function fromProtoBinary(data: ArrayBuffer | Uint8Array): OMLEModel {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const ModelType = getRoot().lookupType('omle.OMLEModel');
  const decoded = ModelType.decode(bytes);
  const obj = ModelType.toObject(decoded, {
    longs: Number,
    enums: String,
    bytes: String,   // base64 — matches the JSON convention
    defaults: false,
  });
  return fromJSON(obj);
}
