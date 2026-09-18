// Unit tests for engine/preprocess.ts — text and numeric preprocessing operators.
// Each test builds a minimal Node + namespace and checks the output tensor.

import { describe, test, expect } from 'vitest';
import {
  executeTokenizer,
  executeRegexTokenizer,
  executeNGram,
  executeStopWordsRemover,
  executeCountVectorizer,
  executeHashingVectorizer,
  executeTfIdfTransformer,
  executeBinarizer,
  executeMinMaxScaler,
  executeNormalizer,
  executeOneHotEncoder,
} from '../../../src/engine/preprocess.js';
import type { Node } from '../../../src/ir.js';
import type { TensorData } from '../../../src/engine/ops.js';
import type { ResolvedModel } from '../../../src/resolve.js';

// ── Test helpers ──────────────────────────────────────────────────────────────

function emptyResolved(): ResolvedModel {
  return { model: { inputs: [] }, tensorIndex: new Map(), executionOrder: [], schemaNames: new Set() };
}

/** Build a minimal Node with the given attributes and one output named "out". */
function node(attrs: Array<{ name: string } & Partial<import('../../../src/ir.js').Attribute>>): Node {
  return {
    name: 'test',
    op: 'test',
    outputs: [{ name: 'out' }],
    attributes: attrs as import('../../../src/ir.js').Attribute[],
  };
}

function stringTd(rows: string[][]): TensorData {
  const maxLen = rows.reduce((m, r) => Math.max(m, r.length), 0) || 1;
  const data: string[] = [];
  for (const r of rows) {
    for (let i = 0; i < maxLen; i++) data.push(r[i] ?? '');
  }
  return { dtype: 'STRING', shape: [rows.length, maxLen], data };
}

function floatTd(values: number[], shape: number[]): TensorData {
  return { dtype: 'FLOAT64', shape, data: new Float64Array(values) };
}

function getStringOut(ns: Map<string, TensorData>): string[][] {
  const td = ns.get('out')!;
  const data = td.data as string[];
  const N = td.shape[0]!;
  const cols = td.shape.length >= 2 ? td.shape[1]! : 1;
  return Array.from({ length: N }, (_, r) =>
    Array.from({ length: cols }, (_, c) => data[r * cols + c] ?? '').filter(t => t !== '')
  );
}

function getFloatOut(ns: Map<string, TensorData>): number[] {
  return Array.from((ns.get('out')!.data) as Float64Array);
}

// ── executeTokenizer ──────────────────────────────────────────────────────────

describe('executeTokenizer', () => {
  test('splits on whitespace and lowercases', () => {
    const ns = new Map<string, TensorData>([
      ['text', { dtype: 'STRING', shape: [2], data: ['Hello World', 'FOO  BAR'] }],
    ]);
    executeTokenizer(node([]), ['text'], ns, 2);
    const out = getStringOut(ns);
    expect(out[0]).toEqual(['hello', 'world']);
    expect(out[1]).toEqual(['foo', 'bar']);
  });

  test('empty string produces empty token list', () => {
    const ns = new Map<string, TensorData>([
      ['text', { dtype: 'STRING', shape: [1], data: [''] }],
    ]);
    executeTokenizer(node([]), ['text'], ns, 1);
    expect(getStringOut(ns)[0]).toEqual([]);
  });
});

// ── executeRegexTokenizer ─────────────────────────────────────────────────────

describe('executeRegexTokenizer', () => {
  test('gaps=true (delimiter pattern) splits on match', () => {
    const ns = new Map<string, TensorData>([
      ['text', { dtype: 'STRING', shape: [1], data: ['a,b,,c'] }],
    ]);
    executeRegexTokenizer(node([
      { name: 'pattern', s: ',' },
      { name: 'gaps', b: true },
      { name: 'min_token_length', i: 1 },
    ]), ['text'], ns, 1);
    // Splitting 'a,b,,c' on ',' → ['a','b','','c']; min_token_length=1 filters ''
    const out = getStringOut(ns)[0];
    expect(out).toEqual(['a', 'b', 'c']);
  });

  test('gaps=false (token pattern) extracts matches', () => {
    const ns = new Map<string, TensorData>([
      ['text', { dtype: 'STRING', shape: [1], data: ['hello world 123'] }],
    ]);
    executeRegexTokenizer(node([
      { name: 'pattern', s: '[a-z]+' },
      { name: 'gaps', b: false },
      { name: 'min_token_length', i: 1 },
    ]), ['text'], ns, 1);
    expect(getStringOut(ns)[0]).toEqual(['hello', 'world']);
  });
});

// ── executeNGram ──────────────────────────────────────────────────────────────

describe('executeNGram', () => {
  test('bigrams from 3 tokens', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['a', 'b', 'c']])],
    ]);
    executeNGram(node([{ name: 'n_min', i: 2 }, { name: 'n_max', i: 2 }]), ['tokens'], ns, 1);
    expect(getStringOut(ns)[0]).toEqual(['a b', 'b c']);
  });

  test('unigrams + bigrams (n_min=1, n_max=2)', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['x', 'y']])],
    ]);
    executeNGram(node([{ name: 'n_min', i: 1 }, { name: 'n_max', i: 2 }]), ['tokens'], ns, 1);
    expect(getStringOut(ns)[0]).toEqual(['x', 'y', 'x y']);
  });

  test('not enough tokens for bigram → empty', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['solo']])],
    ]);
    executeNGram(node([{ name: 'n_min', i: 2 }, { name: 'n_max', i: 2 }]), ['tokens'], ns, 1);
    expect(getStringOut(ns)[0]).toEqual([]);
  });
});

// ── executeStopWordsRemover ───────────────────────────────────────────────────

describe('executeStopWordsRemover', () => {
  test('removes stop words (case-insensitive by default)', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['The', 'cat', 'sat']])],
    ]);
    executeStopWordsRemover(
      node([{ name: 'stop_words', strings: ['the', 'a'] }]),
      ['tokens'], ns, 1, emptyResolved(),
    );
    expect(getStringOut(ns)[0]).toEqual(['cat', 'sat']);
  });

  test('case_sensitive=true keeps mismatched case', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['The', 'the', 'cat']])],
    ]);
    executeStopWordsRemover(
      node([{ name: 'stop_words', strings: ['the'] }, { name: 'case_sensitive', b: true }]),
      ['tokens'], ns, 1, emptyResolved(),
    );
    // 'The' (uppercase) is kept; 'the' is removed
    expect(getStringOut(ns)[0]).toEqual(['The', 'cat']);
  });
});

// ── executeCountVectorizer ────────────────────────────────────────────────────

describe('executeCountVectorizer', () => {
  test('counts token occurrences per vocab term', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['cat', 'dog', 'cat']])],
    ]);
    executeCountVectorizer(
      node([{ name: 'vocabulary', strings: ['cat', 'dog', 'bird'] }]),
      ['tokens'], ns, 1, emptyResolved(),
    );
    // cat=2, dog=1, bird=0
    expect(getFloatOut(ns)).toEqual([2, 1, 0]);
  });

  test('binary=true caps counts at 1', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['a', 'a', 'b']])],
    ]);
    executeCountVectorizer(
      node([{ name: 'vocabulary', strings: ['a', 'b'] }, { name: 'binary', b: true }]),
      ['tokens'], ns, 1, emptyResolved(),
    );
    expect(getFloatOut(ns)).toEqual([1, 1]);
  });

  test('unknown tokens are ignored', () => {
    const ns = new Map<string, TensorData>([
      ['tokens', stringTd([['zzz', 'cat']])],
    ]);
    executeCountVectorizer(
      node([{ name: 'vocabulary', strings: ['cat'] }]),
      ['tokens'], ns, 1, emptyResolved(),
    );
    expect(getFloatOut(ns)).toEqual([1]);
  });
});

// ── executeHashingVectorizer ──────────────────────────────────────────────────

describe('executeHashingVectorizer', () => {
  test('same token always hashes to same bucket', () => {
    const ns1 = new Map<string, TensorData>([['t', stringTd([['hello']])]]);
    const ns2 = new Map<string, TensorData>([['t', stringTd([['hello']])]]);
    const n = node([{ name: 'num_features', i: 16 }]);
    executeHashingVectorizer(n, ['t'], ns1, 1);
    executeHashingVectorizer(n, ['t'], ns2, 1);
    expect(Array.from(ns1.get('out')!.data as Float64Array))
      .toEqual(Array.from(ns2.get('out')!.data as Float64Array));
  });

  test('num_features controls output width', () => {
    const ns = new Map<string, TensorData>([['t', stringTd([['foo', 'bar']])]]);
    executeHashingVectorizer(node([{ name: 'num_features', i: 8 }]), ['t'], ns, 1);
    expect((ns.get('out')!.data as Float64Array).length).toBe(8);
  });

  test('binary=true caps bucket values at 1', () => {
    // Same token twice — count=2 without binary, 1 with binary
    const ns = new Map<string, TensorData>([['t', stringTd([['same', 'same']])]]);
    executeHashingVectorizer(
      node([{ name: 'num_features', i: 16 }, { name: 'binary', b: true }]),
      ['t'], ns, 1,
    );
    const out = Array.from(ns.get('out')!.data as Float64Array);
    expect(out.every(v => v === 0 || v === 1)).toBe(true);
  });
});

// ── executeTfIdfTransformer ───────────────────────────────────────────────────

describe('executeTfIdfTransformer', () => {
  test('tf * idf element-wise', () => {
    // tf = [2, 3]; idf = [0.5, 2.0] → result = [1.0, 6.0]
    const ns = new Map<string, TensorData>([
      ['tf', floatTd([2, 3], [1, 2])],
    ]);
    executeTfIdfTransformer(
      node([{ name: 'idf', tensor: { type: { dtype: 'FLOAT64', shape: [2] }, float64_data: [0.5, 2.0] } }]),
      ['tf'], ns, 1, emptyResolved(),
    );
    const out = getFloatOut(ns);
    expect(out[0]).toBeCloseTo(1.0);
    expect(out[1]).toBeCloseTo(6.0);
  });
});

// ── executeBinarizer ──────────────────────────────────────────────────────────

describe('executeBinarizer', () => {
  test('values above threshold become 1, at or below become 0', () => {
    const ns = new Map<string, TensorData>([
      ['x', floatTd([0, 0.5, 1.0, 2.0], [1, 4])],
    ]);
    // Default threshold = 0: x > 0 → 1
    executeBinarizer(node([{ name: 'threshold', f64: 0.5 }]), ['x'], ns, 1, emptyResolved());
    expect(getFloatOut(ns)).toEqual([0, 0, 1, 1]);
  });
});

// ── executeMinMaxScaler ───────────────────────────────────────────────────────

describe('executeMinMaxScaler', () => {
  test('scales [0,10] to [0,1] with data_min=0, data_range=10', () => {
    const ns = new Map<string, TensorData>([
      ['x', floatTd([0, 5, 10], [1, 3])],
    ]);
    const resolved = emptyResolved();
    executeMinMaxScaler(node([
      { name: 'data_min', tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [0, 0, 0] } },
      { name: 'data_max', tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [10, 10, 10] } },
    ]), ['x'], ns, 1, resolved);
    const out = getFloatOut(ns);
    expect(out[0]).toBeCloseTo(0);
    expect(out[1]).toBeCloseTo(0.5);
    expect(out[2]).toBeCloseTo(1.0);
  });

  test('feature_range_min/max shifts output range', () => {
    const ns = new Map<string, TensorData>([['x', floatTd([0, 10], [1, 2])]]);
    executeMinMaxScaler(node([
      { name: 'data_min', tensor: { type: { dtype: 'FLOAT64', shape: [2] }, float64_data: [0, 0] } },
      { name: 'data_max', tensor: { type: { dtype: 'FLOAT64', shape: [2] }, float64_data: [10, 10] } },
      { name: 'feature_range_min', f64: -1 },
      { name: 'feature_range_max', f64: 1 },
    ]), ['x'], ns, 1, emptyResolved());
    const out = getFloatOut(ns);
    expect(out[0]).toBeCloseTo(-1);   // 0 → -1
    expect(out[1]).toBeCloseTo(1);    // 10 → 1
  });
});

// ── executeNormalizer ─────────────────────────────────────────────────────────

describe('executeNormalizer', () => {
  test('l2: row vector has unit length', () => {
    const ns = new Map<string, TensorData>([['x', floatTd([3, 4], [1, 2])]]);
    executeNormalizer(node([{ name: 'norm', s: 'l2' }]), ['x'], ns, 1);
    const out = getFloatOut(ns);
    const len = Math.sqrt(out[0] ** 2 + out[1] ** 2);
    expect(len).toBeCloseTo(1);
    expect(out[0]).toBeCloseTo(0.6);
    expect(out[1]).toBeCloseTo(0.8);
  });

  test('l1: row sum of abs values equals 1', () => {
    const ns = new Map<string, TensorData>([['x', floatTd([1, 3], [1, 2])]]);
    executeNormalizer(node([{ name: 'norm', s: 'l1' }]), ['x'], ns, 1);
    const out = getFloatOut(ns);
    expect(Math.abs(out[0]) + Math.abs(out[1])).toBeCloseTo(1);
  });

  test('max: max absolute value equals 1', () => {
    const ns = new Map<string, TensorData>([['x', floatTd([2, 8], [1, 2])]]);
    executeNormalizer(node([{ name: 'norm', s: 'max' }]), ['x'], ns, 1);
    const out = getFloatOut(ns);
    expect(Math.max(...out.map(Math.abs))).toBeCloseTo(1);
    expect(out[1]).toBeCloseTo(1);
    expect(out[0]).toBeCloseTo(0.25);
  });

  test('zero vector stays zero (no NaN)', () => {
    const ns = new Map<string, TensorData>([['x', floatTd([0, 0], [1, 2])]]);
    executeNormalizer(node([{ name: 'norm', s: 'l2' }]), ['x'], ns, 1);
    const out = getFloatOut(ns);
    expect(out.every(v => v === 0)).toBe(true);
  });
});

// ── executeOneHotEncoder ──────────────────────────────────────────────────────

describe('executeOneHotEncoder', () => {
  // The engine uses matchCategory which converts the input float to its integer
  // string form ('0', '1', ...) and looks it up in the categories array.
  // Categories must therefore be stored as their string indices ('0','1','2',...).
  // This mirrors the Spark OHE path (after StringIndexer converts strings → floats).

  test('encodes a single categorical feature via numeric indices', () => {
    // input: rows [0, 1, 0] → categories=['0','1','2'] → [1,0,0], [0,1,0], [1,0,0]
    const ns = new Map<string, TensorData>([
      ['x', floatTd([0, 1, 0], [3, 1])],
    ]);
    executeOneHotEncoder(node([
      { name: 'categories', strings: ['0', '1', '2'] },
    ]), ['x'], ns, 3, emptyResolved());
    const out = getFloatOut(ns);
    // row 0: idx=0 → [1,0,0]  row 1: idx=1 → [0,1,0]  row 2: idx=0 → [1,0,0]
    expect(out).toEqual([1, 0, 0,  0, 1, 0,  1, 0, 0]);
  });

  test('unknown category (no matching index) produces all-zero row', () => {
    // val=-1 has no string match in ['0','1'] → all zeros
    const ns = new Map<string, TensorData>([
      ['x', floatTd([-1], [1, 1])],
    ]);
    executeOneHotEncoder(node([
      { name: 'categories', strings: ['0', '1'] },
    ]), ['x'], ns, 1, emptyResolved());
    expect(getFloatOut(ns)).toEqual([0, 0]);
  });
});
