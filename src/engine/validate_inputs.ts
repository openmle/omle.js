import type { OMLEModel, InputSpec, Feature, TensorType } from '../ir.js';

export interface InputIssue {
  kind: 'missing' | 'wrong_type' | 'wrong_shape';
  name: string;
  expected?: string;
  got?: string;
  message: string;
}

export interface InputValidationResult {
  valid: boolean;
  issues: InputIssue[];
}

/**
 * Validates that `inputs` satisfies the model's declared input contract.
 *
 * Checks (in order, per input):
 *   1. Presence — every name in model.inputs (or model_schema.features as fallback) must be present.
 *   2. dtype compatibility — the JS type of the value must be compatible with the declared dtype.
 *   3. Shape — rank must match; fixed dimensions (> 0) must match exactly; ≤ 0 is treated as dynamic.
 */
export function validateInputs(
  model: OMLEModel,
  inputs: Record<string, unknown>,
): InputValidationResult {
  const issues: InputIssue[] = [];
  const specs = buildInputSpecs(model);

  if (specs.length === 0) {
    return { valid: true, issues: [] };
  }

  for (const spec of specs) {
    const provided = Object.prototype.hasOwnProperty.call(inputs, spec.name)
      ? inputs[spec.name]
      : undefined;

    if (provided === undefined || provided === null) {
      issues.push({
        kind:     'missing',
        name:     spec.name,
        expected: spec.typeHint,
        message:  spec.typeHint
          ? `Missing input "${spec.name}" (expected ${spec.typeHint})`
          : `Missing input "${spec.name}"`,
      });
      continue;
    }

    if (spec.dtype) {
      const issue = checkDtype(spec.name, provided, spec.dtype);
      if (issue) { issues.push(issue); continue; }
    }

    if (spec.shape) {
      const issue = checkShape(spec.name, provided, spec.shape, spec.typeHint);
      if (issue) issues.push(issue);
    }
  }

  return { valid: issues.length === 0, issues };
}

// ── Spec building ────────────────────────────────────────────────────────────��

interface SpecEntry {
  name: string;
  dtype?: string;
  shape?: number[];
  typeHint?: string;
}

function buildInputSpecs(model: OMLEModel): SpecEntry[] {
  if (model.inputs && model.inputs.length > 0) {
    return model.inputs.map(specFromInputSpec);
  }
  const features = model.model_schema?.features ?? [];
  const named = features.filter((f): f is Feature & { name: string } =>
    typeof f.name === 'string' && f.name.length > 0,
  );
  return named.map(specFromFeature);
}

function specFromInputSpec(inp: InputSpec): SpecEntry {
  return { name: inp.name, dtype: inp.type?.dtype, shape: normShape(inp.type), typeHint: typeHint(inp.type) };
}

function specFromFeature(feat: Feature & { name: string }): SpecEntry {
  return { name: feat.name, dtype: feat.type?.dtype, shape: normShape(feat.type), typeHint: typeHint(feat.type) };
}

// ── Shape normalization ──────────────────────────────────���────────────────────

// JSON may encode int64 shape dims as strings (e.g. "-1"), coerce them to numbers.
function normShape(t?: TensorType): number[] | undefined {
  if (!t?.shape || t.shape.length === 0) return undefined;
  return t.shape.map(d => Number(d));
}

function typeHint(t?: TensorType): string | undefined {
  if (!t) return undefined;
  const shape = t.shape && t.shape.length > 0
    ? `[${t.shape.map(d => (Number(d) <= 0 ? 'N' : Number(d))).join('×')}]`
    : undefined;
  if (t.dtype && shape) return `${t.dtype}${shape}`;
  if (t.dtype) return t.dtype;
  return shape;
}

// ── Shape inference ───────────────────────────────────────────────────────────

function inferShape(v: unknown): number[] | null {
  if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') return [];

  if (v instanceof Float32Array || v instanceof Float64Array ||
      v instanceof Int8Array    || v instanceof Int16Array  ||
      v instanceof Int32Array   || v instanceof Uint8Array  ||
      v instanceof Uint16Array  || v instanceof Uint32Array) {
    return [v.length];
  }

  if (Array.isArray(v)) {
    if (v.length === 0) return [0];
    const inner = v[0];
    if (Array.isArray(inner)) return [v.length, inner.length];
    return [v.length];
  }

  if (typeof v === 'object' && v !== null) {
    const obj = v as Record<string, unknown>;
    if (Array.isArray(obj['shape'])) return (obj['shape'] as number[]).map(Number);
  }

  return null;
}

// ── Shape check ──────────────────────────────────────────────────────��────────

function checkShape(
  name: string,
  value: unknown,
  expectedShape: number[],
  typeHint?: string,
): InputIssue | null {
  const actualShape = inferShape(value);
  if (actualShape === null) return null;

  // A scalar (actualShape=[]) is compatible with any 1-D expected shape where
  // every fixed dim is 1 or dynamic — e.g. expected [-1] or [1].
  if (actualShape.length === 0) {
    const ok = expectedShape.every(d => Number(d) <= 0 || Number(d) === 1);
    if (ok) return null;
  }

  // When the expected shape's first dim is dynamic (-1/0), a scalar or 1-D
  // array can represent a single-row batch — allow rank-1 vs rank-2 mismatch
  // only when the batch dim is dynamic and remaining dims match.
  const expected = expectedShape.map(Number);
  const isDynamicBatch = expected[0] <= 0;

  if (actualShape.length !== expected.length) {
    // Tolerate: actual=[k] vs expected=[-1, k]  (single-row batch as flat array)
    if (isDynamicBatch && actualShape.length === expected.length - 1) {
      const inner = expected.slice(1);
      const mismatch = inner.findIndex((d, i) => d > 0 && d !== actualShape[i]);
      if (mismatch === -1) return null;
    }

    const expStr = shapeStr(expected);
    const gotStr = shapeStr(actualShape);
    return {
      kind:     'wrong_shape',
      name,
      expected: typeHint ?? expStr,
      got:      gotStr,
      message:  `Input "${name}": rank mismatch — expected ${expStr}, got ${gotStr}`,
    };
  }

  // Same rank — check each fixed dimension.
  for (let i = 0; i < expected.length; i++) {
    const expDim = expected[i];
    const actDim = actualShape[i];
    if (expDim > 0 && actDim !== expDim) {
      const expStr = shapeStr(expected);
      const gotStr = shapeStr(actualShape);
      return {
        kind:     'wrong_shape',
        name,
        expected: typeHint ?? expStr,
        got:      gotStr,
        message:  `Input "${name}": shape mismatch at dim ${i} — expected ${expStr}, got ${gotStr}`,
      };
    }
  }

  return null;
}

function shapeStr(shape: number[]): string {
  if (shape.length === 0) return 'scalar';
  return `[${shape.map(d => (d <= 0 ? 'N' : d)).join('×')}]`;
}

// ── Dtype check ───────────────────────────────────────────────────────────────

const NUMERIC_DTYPES = new Set([
  'BOOL',
  'INT8', 'INT16', 'INT32', 'INT64',
  'UINT8', 'UINT16', 'UINT32', 'UINT64',
  'FLOAT16', 'FLOAT32', 'FLOAT64',
  'DATE', 'TIME', 'TIMESTAMP',
]);

function checkDtype(name: string, value: unknown, dtype: string): InputIssue | null {
  const expectsString  = dtype === 'STRING' || dtype === 'BYTES';
  const expectsNumeric = NUMERIC_DTYPES.has(dtype);

  const jsType = inferJsType(value);

  if (expectsString && jsType !== 'string' && jsType !== 'string[]' && jsType !== 'TensorData') {
    return {
      kind:    'wrong_type',
      name,
      expected: dtype,
      got:     jsType,
      message: `Input "${name}": expected string value for dtype ${dtype}, got ${jsType}`,
    };
  }

  if (expectsNumeric && jsType === 'string') {
    return {
      kind:    'wrong_type',
      name,
      expected: dtype,
      got:     jsType,
      message: `Input "${name}": expected numeric value for dtype ${dtype}, got string`,
    };
  }

  return null;
}

function inferJsType(v: unknown): string {
  if (typeof v === 'string')  return 'string';
  if (typeof v === 'boolean') return 'boolean';
  if (typeof v === 'number')  return 'numeric';
  if (v instanceof Float32Array || v instanceof Float64Array ||
      v instanceof Int32Array   || v instanceof Int8Array   ||
      v instanceof Uint8Array   || v instanceof Uint16Array ||
      v instanceof Uint32Array  || v instanceof BigInt64Array) return 'numeric';
  if (Array.isArray(v)) {
    if (v.length === 0) return 'array';
    return inferJsType(v[0]) + '[]';
  }
  if (typeof v === 'object' && v !== null) {
    const obj = v as Record<string, unknown>;
    if ('dtype' in obj && 'data' in obj) return 'TensorData';
    // IR dense tensor format: {string_data, float64_data, ...}
    if ('string_data' in obj) return 'string[]';
    if ('float64_data' in obj || 'float32_data' in obj ||
        'int32_data'  in obj || 'int64_data'  in obj ||
        'bool_data'   in obj || 'raw_data'    in obj) return 'numeric[]';
    return 'object';
  }
  return typeof v;
}
