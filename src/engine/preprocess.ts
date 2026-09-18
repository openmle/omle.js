// Preprocessing operator implementations for omle.feature and omle.core ops.

import type { Node, Attribute, Expression } from '../ir.js';
import type { TensorData } from './ops.js';
import { tensorToData, tensorCols } from './ops.js';
import type { ResolvedModel } from '../resolve.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function getAttr(node: Node, name: string): Attribute | undefined {
  return (node.attributes ?? []).find(a => a.name === name);
}

function getTensorData(
  attr: Attribute | undefined,
  resolved: ResolvedModel,
): TensorData | null {
  if (!attr) return null;
  if (attr.tensor) return tensorToData(attr.tensor);
  if (attr.tensor_ref) {
    const entry = resolved.tensorIndex.get(attr.tensor_ref.id);
    if (entry?.dense) return tensorToData(entry.dense);
  }
  return null;
}

function getMatrixInput(
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): { data: Float64Array; cols: number } {
  // Merge all inputs into a single [N, totalCols] matrix
  let totalCols = 0;
  for (const name of inputNames) {
    const td = namespace.get(name);
    totalCols += td ? tensorCols(td) : 0;
  }
  const data = new Float64Array(N * totalCols);
  let colOffset = 0;
  for (const name of inputNames) {
    const td = namespace.get(name);
    if (!td) continue;
    const cols = tensorCols(td);
    const src = td.data as Float64Array;
    for (let row = 0; row < N; row++) {
      for (let c = 0; c < cols; c++) {
        data[row * totalCols + colOffset + c] = src[row * cols + c];
      }
    }
    colOffset += cols;
  }
  return { data, cols: totalCols };
}

function publishMatrix(
  node: Node,
  data: Float64Array,
  N: number,
  cols: number,
  namespace: Map<string, TensorData>,
): void {
  const outputs = node.outputs ?? [];
  if (outputs.length >= cols && cols > 1) {
    // Multi-output: publish each column to its own named output
    for (let ci = 0; ci < cols; ci++) {
      const col = new Float64Array(N);
      for (let row = 0; row < N; row++) col[row] = data[row * cols + ci];
      const outName = outputs[ci]?.name;
      if (outName) namespace.set(outName, { dtype: 'FLOAT64', shape: [N], data: col });
    }
  } else {
    const out = outputs[0];
    if (out) namespace.set(out.name, { dtype: 'FLOAT64', shape: cols === 1 ? [N] : [N, cols], data });
  }
}

// ── omle.text ─────────────────────────────────────────────────────────────

export function executeTokenizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const strings = td.data as string[];

  const tokenized = strings.map(s => s.toLowerCase().trim().split(/\s+/).filter(t => t.length > 0));
  const maxLen = tokenized.reduce((m, toks) => Math.max(m, toks.length), 0);

  const out: string[] = [];
  for (const toks of tokenized) {
    for (let i = 0; i < maxLen; i++) out.push(toks[i] ?? '');
  }

  const outEntry = (node.outputs ?? [])[0];
  if (outEntry) {
    namespace.set(outEntry.name, {
      dtype: 'STRING',
      shape: maxLen === 1 ? [N] : [N, maxLen],
      data: out,
    });
  }
}

export function executeRegexTokenizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const strings = td.data as string[];

  const pattern   = (node.attributes ?? []).find(a => a.name === 'pattern')?.s ?? '\\s+';
  const gaps      = (node.attributes ?? []).find(a => a.name === 'gaps')?.b ?? true;
  const minLen    = (node.attributes ?? []).find(a => a.name === 'min_token_length')?.i ?? 1;
  const re        = new RegExp(pattern);

  const tokenized = strings.map(s => {
    const lower = s.toLowerCase();
    const toks  = gaps ? lower.split(re) : (lower.match(new RegExp(pattern, 'g')) ?? []);
    return toks.filter(t => t.length >= minLen);
  });

  const maxLen = tokenized.reduce((m, toks) => Math.max(m, toks.length), 0);
  const out: string[] = [];
  for (const toks of tokenized) {
    for (let i = 0; i < maxLen; i++) out.push(toks[i] ?? '');
  }

  const outEntry = (node.outputs ?? [])[0];
  if (outEntry) {
    namespace.set(outEntry.name, {
      dtype: 'STRING',
      shape: maxLen === 1 ? [N] : [N, maxLen],
      data: out,
    });
  }
}

// Unpack a STRING tensor [N] or [N, M] into per-row token arrays (empty strings filtered).
function getStringRows(td: TensorData, N: number): string[][] {
  const data = td.data as string[];
  const cols = td.shape.length >= 2 ? (td.shape[1] ?? 1) : 1;
  const rows: string[][] = [];
  for (let row = 0; row < N; row++) {
    const toks: string[] = [];
    for (let c = 0; c < cols; c++) {
      const t = data[row * cols + c] ?? '';
      if (t.length > 0) toks.push(t);
    }
    rows.push(toks);
  }
  return rows;
}

function publishStringRows(
  node: Node,
  rows: string[][],
  N: number,
  namespace: Map<string, TensorData>,
): void {
  const maxLen = rows.reduce((m, r) => Math.max(m, r.length), 0);
  const out: string[] = [];
  for (const r of rows) {
    for (let i = 0; i < maxLen; i++) out.push(r[i] ?? '');
  }
  const outEntry = (node.outputs ?? [])[0];
  if (outEntry) {
    namespace.set(outEntry.name, {
      dtype: 'STRING',
      shape: maxLen <= 1 ? [N] : [N, maxLen],
      data: out,
    });
  }
}

export function executeNGram(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const nMin = (node.attributes ?? []).find(a => a.name === 'n_min')?.i ?? 2;
  const nMax = (node.attributes ?? []).find(a => a.name === 'n_max')?.i ?? nMin;
  const rows = getStringRows(td, N);

  const tokenized = rows.map(toks => {
    const ngrams: string[] = [];
    for (let n = nMin; n <= nMax; n++) {
      for (let i = 0; i <= toks.length - n; i++) {
        ngrams.push(toks.slice(i, i + n).join(' '));
      }
    }
    return ngrams;
  });
  publishStringRows(node, tokenized, N, namespace);
}

export function executeStopWordsRemover(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const stopAttr = (node.attributes ?? []).find(a => a.name === 'stop_words');
  const stopTd = getTensorData(stopAttr, resolved);
  const stopWords: string[] = stopTd
    ? (stopTd.data as string[])
    : (stopAttr?.strings ?? []);
  const caseSensitive = (node.attributes ?? []).find(a => a.name === 'case_sensitive')?.b ?? false;

  const stopSet = new Set(caseSensitive ? stopWords : stopWords.map(w => w.toLowerCase()));
  const rows = getStringRows(td, N);
  const filtered = rows.map(toks =>
    toks.filter(t => !stopSet.has(caseSensitive ? t : t.toLowerCase()))
  );
  publishStringRows(node, filtered, N, namespace);
}

export function executeCountVectorizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const vocabAttr = (node.attributes ?? []).find(a => a.name === 'vocabulary');
  const vocabTd = getTensorData(vocabAttr, resolved);
  const vocab: string[] = vocabTd
    ? (vocabTd.data as string[])
    : (vocabAttr?.strings ?? []);
  if (vocab.length === 0) return;
  const binary = (node.attributes ?? []).find(a => a.name === 'binary')?.b ?? false;

  const vocabIndex = new Map<string, number>(vocab.map((w, i) => [w, i]));
  const rows = getStringRows(td, N);
  const out = new Float64Array(N * vocab.length);
  for (let row = 0; row < N; row++) {
    for (const tok of rows[row]) {
      const idx = vocabIndex.get(tok);
      if (idx !== undefined) out[row * vocab.length + idx] += 1;
    }
    if (binary) {
      for (let j = 0; j < vocab.length; j++) {
        if (out[row * vocab.length + j] > 0) out[row * vocab.length + j] = 1;
      }
    }
  }
  publishMatrix(node, out, N, vocab.length, namespace);
}

// MurmurHash3_x86_32 on raw UTF-8 bytes — matches Spark HashingTF (useNewHashingTF=true, seed=42).
function murmur3Hash(s: string): number {
  // Encode string as UTF-8 bytes
  const bytes: number[] = [];
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x80) { bytes.push(c); }
    else if (c < 0x800) { bytes.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f)); }
    else { bytes.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f)); }
  }
  const c1 = 0xcc9e2d51, c2 = 0x1b873593, seed = 42;
  let h1 = seed | 0;
  const nblocks = Math.floor(bytes.length / 4);
  for (let i = 0; i < nblocks; i++) {
    let k1 = (bytes[i*4]) | (bytes[i*4+1] << 8) | (bytes[i*4+2] << 16) | (bytes[i*4+3] << 24);
    k1 = Math.imul(k1, c1) | 0;
    k1 = ((k1 << 15) | (k1 >>> 17)) | 0;
    k1 = Math.imul(k1, c2) | 0;
    h1 ^= k1;
    h1 = ((h1 << 13) | (h1 >>> 19)) | 0;
    h1 = (Math.imul(h1, 5) + 0xe6546b64) | 0;
  }
  let k1 = 0;
  const tail = bytes.length & 3, off = nblocks * 4;
  if (tail >= 3) k1 ^= bytes[off + 2] << 16;
  if (tail >= 2) k1 ^= bytes[off + 1] << 8;
  if (tail >= 1) {
    k1 ^= bytes[off];
    k1 = Math.imul(k1, c1) | 0;
    k1 = ((k1 << 15) | (k1 >>> 17)) | 0;
    k1 = Math.imul(k1, c2) | 0;
    h1 ^= k1;
  }
  h1 ^= bytes.length;
  h1 ^= (h1 >>> 16); h1 = Math.imul(h1, 0x85ebca6b) | 0;
  h1 ^= (h1 >>> 13); h1 = Math.imul(h1, 0xc2b2ae35) | 0;
  h1 ^= (h1 >>> 16);
  return h1;
}

export function executeHashingVectorizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const attrs = node.attributes ?? [];
  const numFeatures = attrs.find(a => a.name === 'num_features')?.i ?? 262144;
  const binary      = attrs.find(a => a.name === 'binary')?.b ?? false;

  const rows = getStringRows(td, N);
  const out  = new Float64Array(N * numFeatures);
  for (let row = 0; row < N; row++) {
    for (const tok of rows[row]) {
      const h   = murmur3Hash(tok);
      const idx = ((h % numFeatures) + numFeatures) % numFeatures;
      if (binary) out[row * numFeatures + idx] = 1;
      else        out[row * numFeatures + idx] += 1;
    }
  }
  publishMatrix(node, out, N, numFeatures, namespace);
}

export function executeTfIdfTransformer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const idfTd = getTensorData((node.attributes ?? []).find(a => a.name === 'idf'), resolved);
  if (!idfTd) return;
  const idf  = idfTd.data as Float64Array;
  const cols = td.shape.length >= 2 ? (td.shape[1] ?? 1) : 1;
  const src  = td.data as Float64Array | Float32Array;
  const out  = new Float64Array(N * cols);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      out[row * cols + c] = Number(src[row * cols + c]) * (idf[c] ?? 0);
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeWord2Vec(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const attrs  = node.attributes ?? [];
  const vocabAttr = attrs.find(a => a.name === 'vocabulary');
  const vocabTd   = getTensorData(vocabAttr, resolved);
  const vocab: string[] = vocabTd ? (vocabTd.data as string[]) : (vocabAttr?.strings ?? []);
  const embTd = getTensorData(attrs.find(a => a.name === 'embeddings'), resolved);
  if (!embTd || vocab.length === 0) return;
  const emb    = embTd.data as Float64Array;
  const embDim = embTd.shape[1] ?? Math.round(emb.length / vocab.length);

  const vocabIndex = new Map<string, number>(vocab.map((w, i) => [w, i]));
  const rows = getStringRows(td, N);
  const out  = new Float64Array(N * embDim);
  for (let row = 0; row < N; row++) {
    const toks = rows[row].filter(t => vocabIndex.has(t));
    if (toks.length === 0) continue;
    for (const tok of toks) {
      const wi = vocabIndex.get(tok)!;
      for (let d = 0; d < embDim; d++) out[row * embDim + d] += emb[wi * embDim + d];
    }
    for (let d = 0; d < embDim; d++) out[row * embDim + d] /= toks.length;
  }
  publishMatrix(node, out, N, embDim, namespace);
}

// ── omle.core ──────────────────────────────────────────────────────────────

export function executeTakeSlots(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  // Name-based selection: look up each named column directly from the namespace.
  // The composite executor populates the namespace with individual schema feature
  // entries so that string-keyed TakeSlots (from ColumnTransformer) can find them.
  const namesAttr = getAttr(node, 'names');
  const names: string[] = namesAttr?.strings ?? [];
  if (names.length > 0) {
    const outCols = names.length;
    const out = new Float64Array(N * outCols);
    for (let ci = 0; ci < outCols; ci++) {
      const td = namespace.get(names[ci]);
      if (!td) continue;
      const src = td.data as Float64Array;
      const srcCols = tensorCols(td);
      for (let row = 0; row < N; row++) {
        out[row * outCols + ci] = src[row * srcCols];
      }
    }
    publishMatrix(node, out, N, outCols, namespace);
    return;
  }

  // Index-based selection.
  const indices: number[] = getAttr(node, 'indices')?.ints ?? [];
  if (indices.length === 0) return;

  const { data: inData, cols: inCols } = getMatrixInput(inputNames, namespace, N);
  const outCols = indices.length;
  const out = new Float64Array(N * outCols);
  for (let row = 0; row < N; row++) {
    for (let ci = 0; ci < outCols; ci++) {
      const srcCol = indices[ci] ?? 0;
      out[row * outCols + ci] = inData[row * inCols + srcCol];
    }
  }
  publishMatrix(node, out, N, outCols, namespace);
}

export function executeConcat(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const { data, cols } = getMatrixInput(inputNames, namespace, N);
  publishMatrix(node, data, N, cols, namespace);
}

export function executeDerive(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const exprAttr = getAttr(node, 'expr');
  if (!exprAttr?.expr) return;

  const expr = exprAttr.expr;

  // Detect simple unary function applied to a matrix reference
  // (e.g. FunctionTransformer log applied to whole matrix X)
  if (expr.apply?.arguments?.length === 1) {
    const arg = expr.apply.arguments[0];
    const fn = expr.apply.function ?? '';
    const baseName = fn.split('.').pop()?.toLowerCase() ?? '';
    const mathFn = resolveMathFn(baseName);
    if (mathFn && arg.ref) {
      const td = namespace.get(arg.ref.value ?? '');
      if (td) {
        const src = td.data as Float64Array;
        const cols = td.shape.length >= 2 ? td.shape.slice(1).reduce((a, b) => a * b, 1) : 1;
        const out = new Float64Array(src.length);
        for (let i = 0; i < src.length; i++) out[i] = mathFn(src[i]);
        publishMatrix(node, out, N, cols, namespace);
        return;
      }
    }
  }

  // General case: evaluate expression per row → single-column output
  const vals: (number | string)[] = new Array(N);
  for (let row = 0; row < N; row++) vals[row] = evalExpr(expr, row, namespace);

  if (N > 0 && typeof vals[0] === 'string') {
    const firstOutput = (node.outputs ?? [])[0];
    if (firstOutput) namespace.set(firstOutput.name, { dtype: 'STRING', shape: [N], data: vals as string[] });
    return;
  }
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) out[i] = vals[i] as number;
  publishMatrix(node, out, N, 1, namespace);
}

function evalExpr(
  expr: Expression,
  row: number,
  namespace: Map<string, TensorData>,
): number | string {
  if (expr.literal != null) {
    const s = expr.literal;
    if (s.double_value !== undefined) return s.double_value;
    if (s.float_value !== undefined) return s.float_value;
    if (s.int_value !== undefined) return s.int_value;
    if (s.string_value !== undefined) return s.string_value;
    return NaN;
  }
  if (expr.ref != null) {
    const name = expr.ref.value ?? '';
    const td = namespace.get(name);
    if (!td) return NaN;
    if (td.dtype === 'STRING') return (td.data as string[])[row] ?? '';
    const data = td.data as Float64Array;
    const cols = td.shape.length >= 2 ? td.shape.slice(1).reduce((a, b) => a * b, 1) : 1;
    return data[row * cols] ?? NaN;
  }
  if (expr.apply != null) {
    const fn = expr.apply.function ?? '';
    const args = expr.apply.arguments ?? [];
    const baseName = fn.split('.').pop()?.toLowerCase() ?? '';

    // Coalesce — variadic, short-circuit on first non-NaN/non-empty
    if (baseName === 'coalesce') {
      for (const arg of args) {
        const v = evalExpr(arg, row, namespace);
        if (typeof v === 'string' || !isNaN(v as number)) return v;
      }
      return NaN;
    }

    // Conditional if — evaluate lazily
    if (baseName === 'if' && args.length === 3) {
      const cond = evalExpr(args[0], row, namespace);
      return (cond !== 0 && !isNaN(cond as number))
        ? evalExpr(args[1], row, namespace)
        : evalExpr(args[2], row, namespace);
    }

    // String operations
    if (baseName === 'lower' && args.length === 1)
      return String(evalExpr(args[0], row, namespace)).toLowerCase();
    if (baseName === 'upper' && args.length === 1)
      return String(evalExpr(args[0], row, namespace)).toUpperCase();
    if (baseName === 'trim' && args.length === 1)
      return String(evalExpr(args[0], row, namespace)).trim();
    if (baseName === 'concat' && args.length === 2)
      return String(evalExpr(args[0], row, namespace)) + String(evalExpr(args[1], row, namespace));
    if (baseName === 'substring' && args.length >= 2) {
      const s = String(evalExpr(args[0], row, namespace));
      const pos = Number(evalExpr(args[1], row, namespace));
      const start = Math.max(0, pos - 1);  // SQL uses 1-based positions
      if (args.length >= 3) {
        const len = Number(evalExpr(args[2], row, namespace));
        return s.slice(start, start + len);
      }
      return s.slice(start);
    }

    // Binary functions
    if (args.length === 2) {
      const xv = evalExpr(args[0], row, namespace);
      const yv = evalExpr(args[1], row, namespace);
      const x = xv as number, y = yv as number;
      switch (baseName) {
        case 'add': case 'plus':                    return x + y;
        case 'subtract': case 'sub': case 'minus':  return x - y;
        case 'multiply': case 'mul':                return x * y;
        case 'divide': case 'div':                  return x / y;
        case 'pow': case 'power':                   return Math.pow(x, y);
        case 'min':                                 return Math.min(x, y);
        case 'max':                                 return Math.max(x, y);
        case 'mod':                                 return x % y;
        case 'equal':                               return (xv === yv) ? 1 : 0;
        case 'not_equal':                           return (xv !== yv) ? 1 : 0;
        case 'less_than':                           return (xv < yv) ? 1 : 0;
        case 'less_or_equal':                       return (xv <= yv) ? 1 : 0;
        case 'greater_than':                        return (xv > yv) ? 1 : 0;
        case 'greater_or_equal':                    return (xv >= yv) ? 1 : 0;
        case 'and':  return (!isNaN(x) && x !== 0 && !isNaN(y) && y !== 0) ? 1 : 0;
        case 'or':   return ((!isNaN(x) && x !== 0) || (!isNaN(y) && y !== 0)) ? 1 : 0;
        default: break;
      }
    }

    // Unary
    if (args.length >= 1) {
      const xv = evalExpr(args[0], row, namespace);
      const x = xv as number;
      const mathFn = resolveMathFn(baseName);
      if (mathFn) return mathFn(x);
      if (baseName === 'not')            return (isNaN(x) || x === 0) ? 1 : 0;
      if (baseName === 'is_missing')     return isNaN(x) ? 1 : 0;
      if (baseName === 'is_not_missing') return isNaN(x) ? 0 : 1;
    }
  }
  return NaN;
}

function resolveMathFn(baseName: string): ((x: number) => number) | null {
  switch (baseName) {
    case 'log': return Math.log;
    case 'log2': return Math.log2;
    case 'log10': return Math.log10;
    case 'log1p': return x => Math.log(1 + x);
    case 'exp': return Math.exp;
    case 'expm1': return x => Math.exp(x) - 1;
    case 'sqrt': return Math.sqrt;
    case 'abs': return Math.abs;
    case 'square': return x => x * x;
    case 'cbrt': return Math.cbrt;
    case 'sign': return Math.sign;
    case 'ceil': return Math.ceil;
    case 'floor': return Math.floor;
    case 'round': return Math.round;
    case 'sigmoid': return x => 1 / (1 + Math.exp(-x));
    case 'tanh': return Math.tanh;
    case 'reciprocal': return x => 1 / x;
    case 'neg': case 'negate': return x => -x;
    default: return null;
  }
}

// ── omle.feature ───────────────────────────────────────────────────────────

export function executeBinarizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const threshAttr = getAttr(node, 'thresholds') ?? getAttr(node, 'threshold');
  const threshTd = threshAttr ? getTensorData(threshAttr, resolved) : null;
  const thresholds: Float64Array | null = threshTd ? (threshTd.data as Float64Array) : null;
  const fallback = threshAttr?.f64 ?? 0;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      const thr = thresholds ? (thresholds[c] ?? thresholds[0] ?? fallback) : fallback;
      out[row * cols + c] = inData[row * cols + c] > thr ? 1 : 0;
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeMinMaxScaler(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const dataMin = getTensorData(getAttr(node, 'data_min'), resolved);
  const dataMax = getTensorData(getAttr(node, 'data_max'), resolved);
  if (!dataMin || !dataMax) return;
  const rangeMin = getAttr(node, 'feature_range_min')?.f64 ?? 0;
  const rangeMax = getAttr(node, 'feature_range_max')?.f64 ?? 1;
  const scale = rangeMax - rangeMin;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const mn = dataMin.data as Float64Array;
  const rng = dataMax.data as Float64Array;  // data_max attr stores data_range_ (max - min)
  const out = new Float64Array(N * cols);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      const x = inData[row * cols + c];
      const r = rng[c] ?? 1;
      const norm = r !== 0 ? (x - (mn[c] ?? 0)) / r : 0;
      out[row * cols + c] = norm * scale + rangeMin;
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeMaxAbsScaler(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const scaleData = getTensorData(getAttr(node, 'scale'), resolved);
  if (!scaleData) return;
  const scale = scaleData.data as Float64Array;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      out[row * cols + c] = inData[row * cols + c] / (scale[c] ?? 1);
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeRobustScaler(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const centerData = getTensorData(getAttr(node, 'center'), resolved);
  const scaleData = getTensorData(getAttr(node, 'scale'), resolved);

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const center = centerData ? centerData.data as Float64Array : null;
  const scale = scaleData ? scaleData.data as Float64Array : null;

  const out = new Float64Array(N * cols);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      let x = inData[row * cols + c];
      if (center) x -= center[c] ?? 0;
      if (scale) x /= (scale[c] ?? 1);
      out[row * cols + c] = x;
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeNormalizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const normType = getAttr(node, 'norm')?.s ?? 'l2';
  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);

  for (let row = 0; row < N; row++) {
    const base = row * cols;
    let norm = 0;
    if (normType === 'l1') {
      for (let c = 0; c < cols; c++) norm += Math.abs(inData[base + c]);
    } else if (normType === 'max') {
      for (let c = 0; c < cols; c++) norm = Math.max(norm, Math.abs(inData[base + c]));
    } else {
      // l2 (default)
      for (let c = 0; c < cols; c++) norm += inData[base + c] ** 2;
      norm = Math.sqrt(norm);
    }
    const inv = norm > 0 ? 1 / norm : 0;
    for (let c = 0; c < cols; c++) out[base + c] = inData[base + c] * inv;
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeOneHotEncoder(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const catAttr = getAttr(node, 'categories');
  const offAttr = getAttr(node, 'category_offsets');
  if (!catAttr) return;

  // Categories may be stored as an inline string list (attr.strings) or as a tensor.
  const catTd = getTensorData(catAttr, resolved);
  const categories: string[] = catTd
    ? (catTd.data as string[])
    : (catAttr.strings ?? []);
  if (categories.length === 0) return;

  // Offsets may be absent (Spark single-feature case) — default to one feature spanning all categories.
  const offTd = offAttr ? getTensorData(offAttr, resolved) : null;
  const offsets: number[] = offTd
    ? Array.from(offTd.data as Float64Array).map(Math.round)
    : [0, categories.length];

  const dropLast = getAttr(node, 'drop_last')?.b ?? false;

  const { data: inData, cols: inCols } = getMatrixInput(inputNames, namespace, N);

  const nFeatures = inCols;
  // Compute per-feature category counts, applying drop_last
  const featCats: number[] = [];
  for (let fi = 0; fi < nFeatures; fi++) {
    const nCats = (offsets[fi + 1] ?? categories.length) - (offsets[fi] ?? 0);
    featCats.push(dropLast ? Math.max(nCats - 1, 0) : nCats);
  }
  const outCols = featCats.reduce((s, c) => s + c, 0);

  const outputs = node.outputs ?? [];
  const onePerOutput = outputs.length >= nFeatures;

  const combined = onePerOutput ? null : new Float64Array(N * outCols);
  let outBase = 0;

  for (let fi = 0; fi < nFeatures; fi++) {
    const start = offsets[fi] ?? 0;
    const end = offsets[fi + 1] ?? categories.length;
    const keep = featCats[fi];

    if (onePerOutput) {
      const col = new Float64Array(N * keep);
      for (let row = 0; row < N; row++) {
        const rawVal = inData[row * nFeatures + fi];
        const catIdx = matchCategory(rawVal, categories, start, end);
        if (catIdx >= 0) {
          const c = catIdx - start;
          if (c < keep) col[row * keep + c] = 1;
        }
      }
      const outName = outputs[fi]?.name;
      if (outName) namespace.set(outName, { dtype: 'FLOAT64', shape: keep === 1 ? [N] : [N, keep], data: col });
    } else {
      for (let row = 0; row < N; row++) {
        const rawVal = inData[row * nFeatures + fi];
        const catIdx = matchCategory(rawVal, categories, start, end);
        if (catIdx >= 0) {
          const c = catIdx - start;
          if (c < keep) combined![row * outCols + outBase + c] = 1;
        }
      }
    }
    outBase += keep;
  }

  if (!onePerOutput) publishMatrix(node, combined!, N, outCols, namespace);
}

function matchCategory(val: number, cats: string[], start: number, end: number): number {
  // Try integer string first, then float string
  const intStr = String(Math.round(val));
  for (let i = start; i < end; i++) {
    if (cats[i] === intStr) return i;
  }
  const floatStr = String(val);
  for (let i = start; i < end; i++) {
    if (cats[i] === floatStr) return i;
  }
  // Try "val.0" form
  const dotStr = val.toFixed(1);
  for (let i = start; i < end; i++) {
    if (cats[i] === dotStr) return i;
  }
  return -1;
}

export function executeBucketizer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const boundAttr = getAttr(node, 'boundaries');
  const offAttr = getAttr(node, 'boundary_offsets');
  if (!boundAttr || !offAttr) return;

  const boundTd = getTensorData(boundAttr, resolved);
  const offTd = getTensorData(offAttr, resolved);
  if (!boundTd || !offTd) return;

  const boundaries = boundTd.data as Float64Array;
  const offsets = Array.from(offTd.data as Float64Array).map(Math.round);

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);

  for (let fi = 0; fi < cols; fi++) {
    const start = offsets[fi] ?? 0;
    const end = offsets[fi + 1] ?? boundaries.length;
    for (let row = 0; row < N; row++) {
      const x = inData[row * cols + fi];
      // bisect_right: count how many boundaries are <= x... actually <= or <
      // sklearn uses digitize which is like searchsorted right
      let bin = 0;
      for (let bi = start; bi < end; bi++) {
        if (x >= (boundaries[bi] ?? Infinity)) bin++;
        else break;
      }
      out[row * cols + fi] = bin;
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executePolynomialFeatures(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const powersAttr = getAttr(node, 'powers');
  if (!powersAttr) return;

  const powersTd = getTensorData(powersAttr, resolved);
  if (!powersTd) return;

  const { data: inData, cols: inCols } = getMatrixInput(inputNames, namespace, N);

  // powers shape: [n_output_features, n_input_features]
  const powersShape = powersTd.shape ?? [];
  const nOut = powersShape[0] ?? 0;
  const nIn = powersShape[1] ?? inCols;
  const powers = powersTd.data as Float64Array;

  const out = new Float64Array(N * nOut);
  for (let row = 0; row < N; row++) {
    for (let oi = 0; oi < nOut; oi++) {
      let val = 1;
      for (let ii = 0; ii < nIn; ii++) {
        const p = powers[oi * nIn + ii] ?? 0;
        if (p !== 0) val *= Math.pow(inData[row * inCols + ii], p);
      }
      out[row * nOut + oi] = val;
    }
  }
  publishMatrix(node, out, N, nOut, namespace);
}

export function executePowerTransformer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const method = getAttr(node, 'method')?.s ?? 'yeo_johnson';
  const standardize = getAttr(node, 'standardize')?.b ?? true;
  const lambdasTd = getTensorData(getAttr(node, 'lambdas'), resolved);
  const meanTd = standardize ? getTensorData(getAttr(node, 'mean'), resolved) : null;
  const scaleTd = standardize ? getTensorData(getAttr(node, 'scale'), resolved) : null;
  if (!lambdasTd) return;

  const lambdas = lambdasTd.data as Float64Array;
  const mean = meanTd ? meanTd.data as Float64Array : null;
  const scale = scaleTd ? scaleTd.data as Float64Array : null;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);

  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      const x = inData[row * cols + c];
      const lam = lambdas[c] ?? 0;
      let y = method === 'box_cox' ? boxCox(x, lam) : yeoJohnson(x, lam);
      if (standardize && mean && scale) {
        y = (y - (mean[c] ?? 0)) / (scale[c] ?? 1);
      }
      out[row * cols + c] = y;
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

function yeoJohnson(x: number, lam: number): number {
  const eps = 1e-10;
  if (x >= 0) {
    if (Math.abs(lam) < eps) return Math.log(x + 1);
    return (Math.pow(x + 1, lam) - 1) / lam;
  } else {
    const lam2 = 2 - lam;
    if (Math.abs(lam2) < eps) return -Math.log(-x + 1);
    return -(Math.pow(-x + 1, lam2) - 1) / lam2;
  }
}

function boxCox(x: number, lam: number): number {
  const eps = 1e-10;
  if (Math.abs(lam) < eps) return Math.log(x);
  return (Math.pow(x, lam) - 1) / lam;
}

export function executeQuantileTransformer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const outputDist = getAttr(node, 'output_distribution')?.s ?? 'uniform';
  const quantilesTd = getTensorData(getAttr(node, 'quantiles'), resolved);
  const referencesTd = getTensorData(getAttr(node, 'references'), resolved);
  if (!quantilesTd || !referencesTd) return;

  // quantiles: [n_quantiles] — the quantile levels (0..1) = t.references_
  // references: [n_features, n_quantiles] — feature values at each level = t.quantiles_.T
  const quantiles = quantilesTd.data as Float64Array;
  const nQ = quantiles.length;
  const references = referencesTd.data as Float64Array;
  const refShape = referencesTd.shape ?? [];
  const nFeatures = refShape[0] ?? 1;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);

  for (let row = 0; row < N; row++) {
    for (let fi = 0; fi < cols && fi < nFeatures; fi++) {
      const x = inData[row * cols + fi];
      const refs = references;
      const fiOffset = fi * nQ;

      // Interpolate x in refs[fi, :] to get quantile level
      let q = interpSearchSorted(refs, fiOffset, nQ, x, quantiles);

      if (outputDist === 'normal') {
        // Clamp slightly away from 0 and 1 to avoid Infinity
        q = Math.max(1e-7, Math.min(1 - 1e-7, q));
        q = probitApprox(q);
      }
      out[row * cols + fi] = q;
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

function interpSearchSorted(
  xp: Float64Array, xpOffset: number, n: number,
  x: number,
  fp: Float64Array,
): number {
  if (n === 0) return 0;
  if (x <= xp[xpOffset]) return fp[0] ?? 0;
  if (x >= xp[xpOffset + n - 1]) return fp[n - 1] ?? 1;
  // Binary search
  let lo = 0, hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (xp[xpOffset + mid] <= x) lo = mid; else hi = mid;
  }
  const x0 = xp[xpOffset + lo], x1 = xp[xpOffset + hi];
  const f0 = fp[lo] ?? 0, f1 = fp[hi] ?? 1;
  const t = x1 !== x0 ? (x - x0) / (x1 - x0) : 0;
  return f0 + t * (f1 - f0);
}

function probitApprox(p: number): number {
  const a = [2.515517, 0.802853, 0.010328];
  const b = [1.432788, 0.189269, 0.001308];
  const sign = p < 0.5 ? -1 : 1;
  const q = Math.min(p, 1 - p);
  const t = Math.sqrt(-2 * Math.log(q));
  const num = a[0] + a[1] * t + a[2] * t * t;
  const den = 1 + b[0] * t + b[1] * t * t + b[2] * t * t * t;
  return sign * (t - num / den);
}

export function executeSplineTransformer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const knotsTd = getTensorData(getAttr(node, 'knots'), resolved);
  const degree = getAttr(node, 'degree')?.i ?? 3;
  const includeBias = getAttr(node, 'include_bias')?.b ?? true;
  const extrapolation = getAttr(node, 'extrapolation')?.s ?? 'constant';
  if (!knotsTd) return;

  // knots shape: [n_aug_knots, n_features]
  const knotsShape = knotsTd.shape ?? [];
  const nAugKnots = knotsShape[0] ?? 0;
  const nFeaturesKnots = knotsShape[1] ?? 1;
  const knotsData = knotsTd.data as Float64Array;

  const nSplines = nAugKnots - degree - 1;  // per feature
  const nSplineOut = includeBias ? nSplines : nSplines - 1;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const outCols = nSplineOut * Math.min(cols, nFeaturesKnots);
  const out = new Float64Array(N * outCols);

  for (let fi = 0; fi < cols && fi < nFeaturesKnots; fi++) {
    // Extract knot vector for feature fi: column fi of knots matrix
    const t = new Float64Array(nAugKnots);
    for (let k = 0; k < nAugKnots; k++) {
      t[k] = knotsData[k * nFeaturesKnots + fi];
    }

    const tMin = t[0], tMax = t[nAugKnots - 1];
    const outOffset = fi * nSplineOut;

    for (let row = 0; row < N; row++) {
      let x = inData[row * cols + fi];

      // Handle extrapolation
      if (extrapolation === 'constant') {
        x = Math.max(tMin, Math.min(tMax, x));
      }

      // Cox-de Boor recursion: compute all B-splines at once
      const basis = bsplineBasis(x, t, nAugKnots, degree, nSplines);

      const splineStart = includeBias ? 0 : 1;
      for (let si = 0; si < nSplineOut; si++) {
        out[row * outCols + outOffset + si] = basis[splineStart + si];
      }
    }
  }
  publishMatrix(node, out, N, outCols, namespace);
}

export function executeNormContinuous(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const origTd = getTensorData(getAttr(node, 'orig_points'), resolved);
  const normTd = getTensorData(getAttr(node, 'norm_points'), resolved);
  const offTd  = getTensorData(getAttr(node, 'point_offsets'), resolved);
  if (!origTd || !normTd) return;

  const orig    = origTd.data as Float64Array;
  const norm    = normTd.data as Float64Array;
  const offsets = offTd ? Array.from(offTd.data as Float64Array).map(Math.round) : [0, orig.length];

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const nFeatures = offsets.length > 1 ? offsets.length - 1 : 1;
  const out = new Float64Array(N * nFeatures);

  for (let fi = 0; fi < nFeatures; fi++) {
    const start = offsets[fi] ?? 0;
    const end   = offsets[fi + 1] ?? orig.length;
    const inCol = Math.min(fi, cols - 1);
    for (let row = 0; row < N; row++) {
      const x = inData[row * cols + inCol];
      let y: number;
      if (x <= orig[start]) {
        y = norm[start];
      } else if (x >= orig[end - 1]) {
        y = norm[end - 1];
      } else {
        // Binary search for the interval
        let lo = start, hi = end - 2;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (orig[mid + 1] <= x) lo = mid + 1; else hi = mid;
        }
        const t = (x - orig[lo]) / (orig[lo + 1] - orig[lo]);
        y = norm[lo] + t * (norm[lo + 1] - norm[lo]);
      }
      out[row * nFeatures + fi] = y;
    }
  }
  publishMatrix(node, out, N, nFeatures, namespace);
}

export function executeWeightedSum(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const weightsTd = getTensorData(getAttr(node, 'weights'), resolved);
  if (!weightsTd) return;
  const weights = weightsTd.data as Float64Array;
  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * cols);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < cols; c++) {
      out[row * cols + c] = inData[row * cols + c] * (weights[c] ?? 1);
    }
  }
  publishMatrix(node, out, N, cols, namespace);
}

export function executeImputer(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const fillTd = getTensorData(getAttr(node, 'fill_tensor'), resolved);
  const fills = fillTd ? (fillTd.data as Float64Array) : new Float64Array(0);
  const outputs = node.outputs ?? [];

  if (inputNames.length === 1) {
    // Single matrix input [N, F] — impute column-wise
    const td = namespace.get(inputNames[0]);
    if (!td) return;
    const src = td.data as Float64Array;
    const cols = td.shape.length >= 2 ? td.shape.slice(1).reduce((a, b) => a * b, 1) : 1;
    const out = new Float64Array(N * cols);
    for (let row = 0; row < N; row++) {
      for (let c = 0; c < cols; c++) {
        const v = src[row * cols + c];
        out[row * cols + c] = isNaN(v) ? (fills[c] ?? 0) : v;
      }
    }
    const outEntry = outputs[0];
    if (outEntry) {
      namespace.set(outEntry.name, {
        dtype: 'FLOAT64',
        shape: cols > 1 ? [N, cols] : [N],
        data: out,
      });
    }
    return;
  }

  // Multiple named inputs — one per feature
  for (let fi = 0; fi < inputNames.length; fi++) {
    const td = namespace.get(inputNames[fi]);
    if (!td) continue;
    const src = td.data as Float64Array;
    const fill = fills[fi] ?? 0;
    const out = new Float64Array(N);
    for (let row = 0; row < N; row++) {
      const v = src[row];
      out[row] = isNaN(v) ? fill : v;
    }
    const outEntry = outputs[fi];
    if (outEntry) namespace.set(outEntry.name, { dtype: 'FLOAT64', shape: [N], data: out });
  }
}

export function executePCA(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const compTd = getTensorData(getAttr(node, 'components'), resolved);
  if (!compTd) return;
  // components shape: [k, n_features]
  const k = compTd.shape[0] ?? 1;
  const nFeatures = compTd.shape[1] ?? (compTd.data.length / k);
  const comp = compTd.data as Float64Array;

  const meanTd = getTensorData(getAttr(node, 'mean'), resolved);
  const mean = meanTd?.data as Float64Array | undefined;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  const out = new Float64Array(N * k);

  for (let row = 0; row < N; row++) {
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let i = 0; i < nFeatures; i++) {
        const x = inData[row * cols + i] - (mean?.[i] ?? 0);
        sum += x * comp[j * nFeatures + i];
      }
      out[row * k + j] = sum;
    }
  }
  publishMatrix(node, out, N, k, namespace);
}

function bsplineBasis(x: number, t: Float64Array, n: number, degree: number, nSplines: number): Float64Array {
  // Cox-de Boor algorithm: compute all B-spline basis values at x
  const b = new Float64Array(n);

  // Degree 0: indicator functions
  for (let i = 0; i < n - 1; i++) {
    b[i] = (x >= t[i] && x < t[i + 1]) ? 1 : 0;
  }
  // Handle right boundary: last spline should be 1 at t[-1]
  if (x === t[n - 1]) b[n - degree - 2] = 1;

  // Build up degrees 1..degree
  for (let d = 1; d <= degree; d++) {
    for (let i = 0; i < n - d - 1; i++) {
      const left = t[i + d] - t[i] > 0
        ? ((x - t[i]) / (t[i + d] - t[i])) * b[i]
        : 0;
      const right = t[i + d + 1] - t[i + 1] > 0
        ? ((t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1])) * b[i + 1]
        : 0;
      b[i] = left + right;
    }
  }

  return b.slice(0, nSplines);
}

// ── TruncatedSVD ──────────────────────────────────────────────────────────────

export function executeTruncatedSVD(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const compData = getTensorData(getAttr(node, 'components'), resolved);
  if (!compData) return;

  // components shape: [n_components, n_features]
  const compShape = compData.shape;
  const nComponents = compShape.length >= 2 ? Number(compShape[0]) : 1;
  const nFeatures   = compShape.length >= 2 ? Number(compShape[1]) : compData.data.length;
  const comp = compData.data as Float64Array;

  const { data: inData, cols } = getMatrixInput(inputNames, namespace, N);
  if (cols !== nFeatures) return;  // shape mismatch

  // X_out = X @ components.T  →  shape [N, nComponents]
  const out = new Float64Array(N * nComponents);
  for (let row = 0; row < N; row++) {
    for (let c = 0; c < nComponents; c++) {
      let acc = 0;
      for (let k = 0; k < nFeatures; k++) {
        acc += inData[row * cols + k] * comp[c * nFeatures + k];
      }
      out[row * nComponents + c] = acc;
    }
  }
  publishMatrix(node, out, N, nComponents, namespace);
}
