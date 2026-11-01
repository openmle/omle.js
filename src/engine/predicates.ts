// Predicate evaluation for the reference engine.
// Predicates are evaluated per-row over named column values.

import type {
  Predicate, SimplePredicate, SimpleSetPredicate, CompoundPredicate, Scalar,
} from '../ir.js';
import { scalarValue } from '../ir.js';

// ── Column resolver ───────────────────────────────────────────────────────────

export type ColumnResolver = (name: string, row: number) => number | string | boolean | undefined;

// ── Entry point ───────────────────────────────────────────────────────────────

export function evalPredicate(
  pred: Predicate,
  resolver: ColumnResolver,
  row: number,
): boolean | null {
  if (pred.true_predicate !== undefined) return true;
  if (pred.false_predicate !== undefined) return false;
  if (pred.simple) return evalSimple(pred.simple, resolver, row);
  if (pred.simple_set) return evalSimpleSet(pred.simple_set, resolver, row);
  if (pred.compound) return evalCompound(pred.compound, resolver, row);
  return null;
}

// ── Simple predicate ──────────────────────────────────────────────────────────

function evalSimple(
  pred: SimplePredicate,
  resolver: ColumnResolver,
  row: number,
): boolean | null {
  const val = resolver(pred.column.value, row);
  const isMissing = val === undefined || val === null || (typeof val === 'number' && isNaN(val));

  switch (pred.op) {
    case 'IS_MISSING': return isMissing;
    case 'IS_NOT_MISSING': return !isMissing;
    default: break;
  }

  if (isMissing) return null;

  const cmp = pred.value ? scalarValue(pred.value) : undefined;
  switch (pred.op) {
    case 'LESS_THAN': return (val as number) < (cmp as number);
    case 'LESS_OR_EQUAL': return (val as number) <= (cmp as number);
    case 'GREATER_THAN': return (val as number) > (cmp as number);
    case 'GREATER_OR_EQUAL': return (val as number) >= (cmp as number);
    case 'EQUAL': return val === cmp;
    case 'NOT_EQUAL': return val !== cmp;
    default: return null;
  }
}

// ── Set predicate ─────────────────────────────────────────────────────────────

function evalSimpleSet(
  pred: SimpleSetPredicate,
  resolver: ColumnResolver,
  row: number,
): boolean | null {
  const val = resolver(pred.column.value, row);
  if (val === undefined || val === null) return null;

  const setVals = new Set((pred.values ?? []).map(s => scalarValue(s)));
  switch (pred.op) {
    case 'IN': return setVals.has(val);
    case 'NOT_IN': return !setVals.has(val);
    default: return null;
  }
}

// ── Compound predicate ────────────────────────────────────────────────────────

function evalCompound(
  pred: CompoundPredicate,
  resolver: ColumnResolver,
  row: number,
): boolean | null {
  const children = pred.predicates ?? [];
  switch (pred.op) {
    case 'AND': {
      let result: boolean | null = true;
      for (const c of children) {
        const r = evalPredicate(c, resolver, row);
        if (r === false) return false;
        if (r === null) result = null;
      }
      return result;
    }
    case 'OR': {
      let result: boolean | null = false;
      for (const c of children) {
        const r = evalPredicate(c, resolver, row);
        if (r === true) return true;
        if (r === null) result = null;
      }
      return result;
    }
    case 'XOR': {
      let trueCount = 0;
      for (const c of children) {
        const r = evalPredicate(c, resolver, row);
        if (r === null) return null;
        if (r) trueCount++;
      }
      return trueCount % 2 === 1;
    }
    case 'SURROGATE': {
      for (const c of children) {
        const r = evalPredicate(c, resolver, row);
        if (r !== null) return r;
      }
      return null;
    }
    default:
      return null;
  }
}

// ── Scalar comparison helpers ─────────────────────────────────────────────────

export function scalarEquals(a: Scalar, b: Scalar): boolean {
  return scalarValue(a) === scalarValue(b);
}
