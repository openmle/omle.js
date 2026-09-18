// Unit tests for engine/predicates.ts — pure predicate logic, no I/O.

import { describe, test, expect } from 'vitest';
import { evalPredicate, scalarEquals } from '../../../src/engine/predicates.js';
import type { Predicate, ColumnResolver } from '../../../src/engine/predicates.js';
import type { Scalar } from '../../../src/ir.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a resolver from a flat row object. */
function resolver(row: Record<string, number | string | boolean | undefined>): ColumnResolver {
  return (name) => row[name];
}

function pred(p: Partial<Predicate>): Predicate {
  return p as Predicate;
}

function scalar(v: number | string | boolean): Scalar {
  if (typeof v === 'number') return { double_value: v };
  if (typeof v === 'string') return { string_value: v };
  return { bool_value: v };
}

// ── TruePredicate / FalsePredicate ────────────────────────────────────────────

describe('TruePredicate / FalsePredicate', () => {
  test('true_predicate always returns true', () => {
    expect(evalPredicate(pred({ true_predicate: {} }), resolver({}), 0)).toBe(true);
  });

  test('false_predicate always returns false', () => {
    expect(evalPredicate(pred({ false_predicate: {} }), resolver({}), 0)).toBe(false);
  });

  test('empty predicate returns null', () => {
    expect(evalPredicate(pred({}), resolver({}), 0)).toBeNull();
  });
});

// ── SimplePredicate ───────────────────────────────────────────────────────────

describe('SimplePredicate — numeric comparisons', () => {
  const col = { value: 'age' };

  test('LESS_THAN', () => {
    const p = pred({ simple: { column: col, op: 'LESS_THAN', value: scalar(30) } });
    expect(evalPredicate(p, resolver({ age: 20 }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ age: 30 }), 0)).toBe(false);
    expect(evalPredicate(p, resolver({ age: 40 }), 0)).toBe(false);
  });

  test('LESS_OR_EQUAL', () => {
    const p = pred({ simple: { column: col, op: 'LESS_OR_EQUAL', value: scalar(30) } });
    expect(evalPredicate(p, resolver({ age: 30 }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ age: 31 }), 0)).toBe(false);
  });

  test('GREATER_THAN', () => {
    const p = pred({ simple: { column: col, op: 'GREATER_THAN', value: scalar(30) } });
    expect(evalPredicate(p, resolver({ age: 31 }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ age: 30 }), 0)).toBe(false);
  });

  test('GREATER_OR_EQUAL', () => {
    const p = pred({ simple: { column: col, op: 'GREATER_OR_EQUAL', value: scalar(30) } });
    expect(evalPredicate(p, resolver({ age: 30 }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ age: 29 }), 0)).toBe(false);
  });

  test('EQUAL / NOT_EQUAL', () => {
    const p = pred({ simple: { column: col, op: 'EQUAL', value: scalar(5) } });
    expect(evalPredicate(p, resolver({ age: 5 }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ age: 6 }), 0)).toBe(false);

    const ne = pred({ simple: { column: col, op: 'NOT_EQUAL', value: scalar(5) } });
    expect(evalPredicate(ne, resolver({ age: 6 }), 0)).toBe(true);
    expect(evalPredicate(ne, resolver({ age: 5 }), 0)).toBe(false);
  });
});

describe('SimplePredicate — missing values', () => {
  const col = { value: 'x' };

  test('IS_MISSING is true when column is undefined', () => {
    const p = pred({ simple: { column: col, op: 'IS_MISSING' } });
    expect(evalPredicate(p, resolver({}), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ x: 1 }), 0)).toBe(false);
  });

  test('IS_NOT_MISSING is true when column has a value', () => {
    const p = pred({ simple: { column: col, op: 'IS_NOT_MISSING' } });
    expect(evalPredicate(p, resolver({ x: 0 }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({}), 0)).toBe(false);
  });

  test('numeric op returns null when column is missing', () => {
    const p = pred({ simple: { column: col, op: 'LESS_THAN', value: scalar(10) } });
    expect(evalPredicate(p, resolver({}), 0)).toBeNull();
  });

  test('IS_MISSING is true for NaN', () => {
    const p = pred({ simple: { column: col, op: 'IS_MISSING' } });
    expect(evalPredicate(p, resolver({ x: NaN }), 0)).toBe(true);
  });
});

// ── SimpleSetPredicate ────────────────────────────────────────────────────────

describe('SimpleSetPredicate', () => {
  const col = { value: 'cat' };
  const values = [scalar('a'), scalar('b'), scalar('c')];

  test('IN returns true when value is in set', () => {
    const p = pred({ simple_set: { column: col, op: 'IN', values } });
    expect(evalPredicate(p, resolver({ cat: 'b' }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ cat: 'd' }), 0)).toBe(false);
  });

  test('NOT_IN returns true when value is not in set', () => {
    const p = pred({ simple_set: { column: col, op: 'NOT_IN', values } });
    expect(evalPredicate(p, resolver({ cat: 'z' }), 0)).toBe(true);
    expect(evalPredicate(p, resolver({ cat: 'a' }), 0)).toBe(false);
  });

  test('returns null when column is missing', () => {
    const p = pred({ simple_set: { column: col, op: 'IN', values } });
    expect(evalPredicate(p, resolver({}), 0)).toBeNull();
  });
});

// ── CompoundPredicate ─────────────────────────────────────────────────────────

describe('CompoundPredicate AND', () => {
  const t = pred({ true_predicate: {} });
  const f = pred({ false_predicate: {} });

  test('AND(true, true) → true', () => {
    const p = pred({ compound: { op: 'AND', predicates: [t, t] } });
    expect(evalPredicate(p, resolver({}), 0)).toBe(true);
  });

  test('AND(true, false) → false (short-circuits)', () => {
    const p = pred({ compound: { op: 'AND', predicates: [t, f] } });
    expect(evalPredicate(p, resolver({}), 0)).toBe(false);
  });

  test('AND with null child → null if no false', () => {
    const nil = pred({});  // evalPredicate returns null
    const p = pred({ compound: { op: 'AND', predicates: [t, nil] } });
    expect(evalPredicate(p, resolver({}), 0)).toBeNull();
  });
});

describe('CompoundPredicate OR', () => {
  const t = pred({ true_predicate: {} });
  const f = pred({ false_predicate: {} });

  test('OR(false, true) → true', () => {
    const p = pred({ compound: { op: 'OR', predicates: [f, t] } });
    expect(evalPredicate(p, resolver({}), 0)).toBe(true);
  });

  test('OR(false, false) → false', () => {
    const p = pred({ compound: { op: 'OR', predicates: [f, f] } });
    expect(evalPredicate(p, resolver({}), 0)).toBe(false);
  });
});

describe('CompoundPredicate XOR', () => {
  const t = pred({ true_predicate: {} });
  const f = pred({ false_predicate: {} });

  test('XOR(true, false) → true (odd count of trues)', () => {
    expect(evalPredicate(pred({ compound: { op: 'XOR', predicates: [t, f] } }), resolver({}), 0)).toBe(true);
  });

  test('XOR(true, true) → false (even count of trues)', () => {
    expect(evalPredicate(pred({ compound: { op: 'XOR', predicates: [t, t] } }), resolver({}), 0)).toBe(false);
  });
});

describe('CompoundPredicate SURROGATE', () => {
  const nil = pred({});  // returns null
  const t   = pred({ true_predicate: {} });
  const f   = pred({ false_predicate: {} });

  test('returns first non-null child result', () => {
    const p = pred({ compound: { op: 'SURROGATE', predicates: [nil, t, f] } });
    expect(evalPredicate(p, resolver({}), 0)).toBe(true);
  });

  test('returns null if all children are null', () => {
    const p = pred({ compound: { op: 'SURROGATE', predicates: [nil, nil] } });
    expect(evalPredicate(p, resolver({}), 0)).toBeNull();
  });
});

// ── scalarEquals ──────────────────────────────────────────────────────────────

describe('scalarEquals', () => {
  test('numeric equality', () => {
    expect(scalarEquals({ double_value: 1.5 }, { double_value: 1.5 })).toBe(true);
    expect(scalarEquals({ double_value: 1.5 }, { double_value: 2.0 })).toBe(false);
  });

  test('string equality', () => {
    expect(scalarEquals({ string_value: 'a' }, { string_value: 'a' })).toBe(true);
    expect(scalarEquals({ string_value: 'a' }, { string_value: 'b' })).toBe(false);
  });

  test('bool equality', () => {
    expect(scalarEquals({ bool_value: true }, { bool_value: true })).toBe(true);
    expect(scalarEquals({ bool_value: true }, { bool_value: false })).toBe(false);
  });
});
