// JSON and binary I/O for OMLE models.
//
// JSON convention (Python reference SDK):
//  - enums as string names ("FLOAT32", not 11)
//  - absent / empty fields omitted
//  - bytes as base64 strings
//  - tensor typed payloads as flat arrays (float32_data: [1,2,3], not {values:[1,2,3]})

import type { OMLEModel } from './ir.js';

// ── JSON ──────────────────────────────────────────────────────────────────────

export function fromJSON(json: string | object): OMLEModel {
  const raw = typeof json === 'string' ? JSON.parse(json) : json;
  return flattenListWrappers(raw) as OMLEModel;
}

// Collapse proto list-wrapper objects ({ values: [...] }) to plain arrays.
// The Python SDK sometimes emits these for repeated scalar fields in JSON.
function flattenListWrappers(val: unknown): unknown {
  if (val === null || val === undefined) return val;
  if (typeof val !== 'object') return val;
  if (Array.isArray(val)) return val.map(flattenListWrappers);
  const obj = val as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length === 1 && keys[0] === 'values' && Array.isArray(obj['values'])) {
    const arr = obj['values'] as unknown[];
    // Only collapse proto scalar list-wrappers (e.g. float32_data: {values: [1,2,3]}).
    // Object arrays like DiscreteDomain.values contain DomainValue messages — keep them structured,
    // or Array.prototype.values would shadow the field and crash callers that do .slice()/.map().
    if (arr.every(item => item === null || typeof item !== 'object')) {
      return arr.map(flattenListWrappers);
    }
  }
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) result[k] = flattenListWrappers(v);
  return result;
}

export function toJSON(model: OMLEModel, pretty = false): string {
  return JSON.stringify(stripEmpty(model as unknown as JsonValue), null, pretty ? 2 : undefined);
}

// ── File helpers (Node.js / browser) ─────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const _g = globalThis as any;
function isNode(): boolean {
  return typeof _g['process'] !== 'undefined' && Boolean(_g['process']?.versions?.node);
}

export async function loadFile(path: string): Promise<OMLEModel> {
  if (!isNode()) throw new Error('loadFile is not available in the browser; use loadBlob instead');
  // Variable import avoids static module resolution for browser build contexts
  const m = 'node:fs/promises';
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fs: any = await import(/* @vite-ignore */ m);
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  if (ext === 'json') return fromJSON(await fs.readFile(path, 'utf8') as string);
  throw new Error(`Unsupported extension: ${ext}. Use loadProtoFile for binary formats.`);
}

export async function loadBlob(blob: { text(): Promise<string> }): Promise<OMLEModel> {
  return fromJSON(await blob.text());
}

export async function saveFile(model: OMLEModel, path: string, pretty = true): Promise<void> {
  if (!isNode()) throw new Error('saveFile is not available in the browser');
  const m = 'node:fs/promises';
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fs: any = await import(/* @vite-ignore */ m);
  await fs.writeFile(path, toJSON(model, pretty), 'utf8');
}

// ── Internal helpers ──────────────────────────────────────────────────────────

type JsonValue = string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue };

function stripEmpty(val: JsonValue): JsonValue {
  if (val === null || val === undefined) return val;
  if (Array.isArray(val)) {
    const arr = val.map(stripEmpty).filter(v => v !== undefined);
    return arr as JsonValue[];
  }
  if (typeof val === 'object') {
    const out: { [k: string]: JsonValue } = {};
    for (const [k, v] of Object.entries(val as { [k: string]: JsonValue })) {
      const sv = stripEmpty(v);
      if (sv !== undefined && sv !== null) {
        if (Array.isArray(sv) && sv.length === 0) continue;
        if (typeof sv === 'object' && !Array.isArray(sv) && Object.keys(sv).length === 0) continue;
        out[k] = sv;
      }
    }
    return out;
  }
  return val;
}
