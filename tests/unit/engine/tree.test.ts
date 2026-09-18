// Unit tests for engine/tree.ts — decision tree and tree ensemble execution.

import { describe, test, expect } from 'vitest';
import { executeTree, executeTreeEnsemble } from '../../../src/engine/tree.js';
import type { Tree, TreeEnsemble } from '../../../src/ir.js';
import type { ResolvedModel } from '../../../src/resolve.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function emptyResolved(): ResolvedModel {
  return { model: { inputs: [] }, tensorIndex: new Map(), executionOrder: [], schemaNames: new Set() };
}

function f64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

function expectClose(actual: Float64Array, expected: number[], tol = 1e-8): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], -Math.log10(tol));
  }
}

// ── Minimal tree builder ──────────────────────────────────────────────────────
//
// Two-node tree (depth-1 stump):
//
//   node 0 (BRANCH): split on feature `feat` < `threshold`
//     left  child (node 1): LEAF → value `leftVal`
//     right child (node 2): LEAF → value `rightVal`
//
// node_kind:        ['BRANCH', 'LEAF', 'LEAF']
// split_feature:    [feat, 0, 0]
// split_op:         ['LESS_THAN', 'LESS_THAN', 'LESS_THAN']
// split_threshold:  [threshold, 0, 0]   (inline float64_data)
// children_offset:  [0, 2, 4]
// children_index:   [1, 2]              (left, right for node 0)
// leaf_value:       [0, leftVal, rightVal]

function stump(feat: number, threshold: number, leftVal: number, rightVal: number): Tree {
  return {
    node_kind:       ['BRANCH', 'LEAF', 'LEAF'],
    split_feature:   [feat, 0, 0],
    split_op:        ['LESS_THAN', 'LESS_THAN', 'LESS_THAN'],
    split_threshold: { tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [threshold, 0, 0] } },
    children_offset: [0, 2, 4],
    children_index:  [1, 2],
    leaf_value:      { tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [0, leftVal, rightVal] } },
  };
}

// ── executeTree ───────────────────────────────────────────────────────────────

describe('executeTree', () => {
  test('single sample goes left when feature < threshold', () => {
    const tree = stump(0, 5.0, 10.0, 20.0);
    const out = executeTree(tree, f64(3.0), 1, 1, emptyResolved());
    expectClose(out, [10.0]);
  });

  test('single sample goes right when feature >= threshold', () => {
    const tree = stump(0, 5.0, 10.0, 20.0);
    const out = executeTree(tree, f64(7.0), 1, 1, emptyResolved());
    expectClose(out, [20.0]);
  });

  test('batch of 3 samples routed independently', () => {
    const tree = stump(0, 5.0, 10.0, 20.0);
    // feature values: [3, 7, 5] → [left, right, right]
    const out = executeTree(tree, f64(3.0, 7.0, 5.0), 1, 3, emptyResolved());
    expectClose(out, [10.0, 20.0, 20.0]);
  });

  test('split on second feature (numSlots=2)', () => {
    const tree = stump(1, 0.0, -1.0, 1.0);  // split on feature index 1
    // rows: [_, 5], [_, -3]  → right, left
    const out = executeTree(tree, f64(0, 5,  0, -3), 2, 2, emptyResolved());
    expectClose(out, [1.0, -1.0]);
  });

  test('NaN feature falls back to default_child', () => {
    const tree: Tree = {
      ...stump(0, 5.0, 10.0, 20.0),
      default_child: [2, 0, 0],  // node 0 missing → go to right child (node 2)
    };
    const out = executeTree(tree, f64(NaN), 1, 1, emptyResolved());
    expectClose(out, [20.0]);
  });

  test('IN_SET split op routes correctly', () => {
    const tree: Tree = {
      node_kind:        ['BRANCH', 'LEAF', 'LEAF'],
      split_feature:    [0, 0, 0],
      split_op:         ['IN_SET', 'LESS_THAN', 'LESS_THAN'],
      split_threshold:  { tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [0, 0, 0] } },
      children_offset:  [0, 2, 4],
      children_index:   [1, 2],
      leaf_value:       { tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [0, 1.0, 2.0] } },
      category_set:     [1, 3],           // set = {1, 3}
      category_set_offset: [0, 2, 2],
      category_set_count:  [2, 0, 0],
    };
    // val=1 → IN_SET → left (1.0); val=2 → not in set → right (2.0)
    const out = executeTree(tree, f64(1.0, 2.0), 1, 2, emptyResolved());
    expectClose(out, [1.0, 2.0]);
  });

  test('leaf_vector (multi-output leaf)', () => {
    // Each leaf yields a 2-element vector
    const tree: Tree = {
      node_kind:       ['BRANCH', 'LEAF', 'LEAF'],
      split_feature:   [0, 0, 0],
      split_op:        ['LESS_THAN', 'LESS_THAN', 'LESS_THAN'],
      split_threshold: { tensor: { type: { dtype: 'FLOAT64', shape: [3] }, float64_data: [5, 0, 0] } },
      children_offset: [0, 2, 4],
      children_index:  [1, 2],
      leaf_width:      2,
      leaf_vector:     { tensor: { type: { dtype: 'FLOAT64', shape: [4] }, float64_data: [0, 0, 0.3, 0.7] } },
      leaf_vector_index: [0, 0, 2],  // node1→offset 0; node2→offset 2
    };
    // feature=3 → left (node 1) → [0, 0]; feature=7 → right (node 2) → [0.3, 0.7]
    const out = executeTree(tree, f64(3.0, 7.0), 1, 2, emptyResolved());
    expectClose(out, [0, 0, 0.3, 0.7]);
  });
});

// ── executeTreeEnsemble ───────────────────────────────────────────────────────

describe('executeTreeEnsemble — SUM aggregation', () => {
  test('two stumps: scores added', () => {
    const t1 = stump(0, 5.0, 1.0, -1.0);  // x<5 → 1, else -1
    const t2 = stump(0, 3.0, 0.5, -0.5); // x<3 → 0.5, else -0.5
    const ensemble: TreeEnsemble = { trees: [t1, t2], aggregation: 'SUM' };
    // x=2: t1→1, t2→0.5 → 1.5; x=4: t1→1, t2→-0.5 → 0.5; x=6: t1→-1, t2→-0.5 → -1.5
    const out = executeTreeEnsemble(ensemble, f64(2.0, 4.0, 6.0), 1, 3, emptyResolved());
    expectClose(out, [1.5, 0.5, -1.5]);
  });

  test('base_score is added to all outputs', () => {
    const ensemble: TreeEnsemble = {
      trees: [stump(0, 5.0, 0.0, 0.0)],
      aggregation: 'SUM',
      base_score: { double_value: 10.0 },
    };
    const out = executeTreeEnsemble(ensemble, f64(3.0), 1, 1, emptyResolved());
    expectClose(out, [10.0]);
  });

  test('tree_weights scale individual tree outputs', () => {
    const t1 = stump(0, 5.0, 2.0, 0.0);
    const t2 = stump(0, 5.0, 3.0, 0.0);
    const ensemble: TreeEnsemble = {
      trees: [t1, t2],
      aggregation: 'WEIGHTED_SUM',
      tree_weights: { tensor: { type: { dtype: 'FLOAT64', shape: [2] }, float64_data: [0.5, 0.5] } },
    };
    // x=3 → t1=2, t2=3; weighted sum = 0.5*2+0.5*3 = 2.5
    const out = executeTreeEnsemble(ensemble, f64(3.0), 1, 1, emptyResolved());
    expectClose(out, [2.5]);
  });
});

describe('executeTreeEnsemble — AVERAGE aggregation', () => {
  test('two stumps: scores averaged', () => {
    const t1 = stump(0, 5.0, 4.0, 0.0);
    const t2 = stump(0, 5.0, 0.0, 0.0);  // always 0 for x<5
    const ensemble: TreeEnsemble = { trees: [t1, t2], aggregation: 'AVERAGE' };
    // x=3: t1=4, t2=0 → avg=2
    const out = executeTreeEnsemble(ensemble, f64(3.0), 1, 1, emptyResolved());
    expectClose(out, [2.0]);
  });
});

describe('executeTreeEnsemble — MAJORITY_VOTE aggregation', () => {
  test('majority label wins (3 trees, 2 vote for label 1)', () => {
    // leftVal=1, rightVal=0: x<threshold → label 1; x>=threshold → label 0
    const t0 = stump(0, 5.0, 1.0, 0.0);  // x<5 → 1 (x=4 → 1)
    const t1 = stump(0, 3.0, 1.0, 0.0);  // x<3 → 1 (x=4 → 0)
    const t2 = stump(0, 7.0, 1.0, 0.0);  // x<7 → 1 (x=4 → 1)
    const ensemble: TreeEnsemble = { trees: [t0, t1, t2], aggregation: 'MAJORITY_VOTE' };
    // x=4: t0→1, t1→0, t2→1 → 2 votes for 1, majority=1
    const out = executeTreeEnsemble(ensemble, f64(4.0), 1, 1, emptyResolved());
    expect(out[0]).toBe(1);
  });
});

describe('executeTreeEnsemble — multiclass with tree_group', () => {
  test('class-grouped trees accumulate per-class scores', () => {
    // 2 classes, 2 trees: tree 0 → class 0, tree 1 → class 1
    const t0 = stump(0, 5.0, 3.0, 0.0);  // class 0 score
    const t1 = stump(0, 5.0, 0.0, 5.0);  // class 1 score: x<5 → 0, else 5
    const ensemble: TreeEnsemble = {
      trees: [t0, t1],
      aggregation: 'SUM',
      tree_group: [0, 1],
    };
    // x=3: t0→3.0 (class 0), t1→0.0 (class 1) → [3, 0]
    const out = executeTreeEnsemble(ensemble, f64(3.0), 1, 1, emptyResolved());
    expect(out.length).toBe(2);
    expectClose(out, [3.0, 0.0]);
  });
});

describe('executeTreeEnsemble — post_transform SIGMOID', () => {
  test('SIGMOID applied to SUM output', () => {
    const ensemble: TreeEnsemble = {
      trees: [stump(0, 5.0, 0.0, 0.0)],  // always 0
      aggregation: 'SUM',
      post_transform: 'SIGMOID',
    };
    const out = executeTreeEnsemble(ensemble, f64(3.0), 1, 1, emptyResolved());
    expect(out[0]).toBeCloseTo(0.5, 8);  // sigmoid(0) = 0.5
  });
});

describe('executeTreeEnsemble — edge cases', () => {
  test('empty trees array returns zeros', () => {
    const ensemble: TreeEnsemble = { trees: [], aggregation: 'SUM' };
    const out = executeTreeEnsemble(ensemble, f64(1.0, 2.0), 1, 2, emptyResolved());
    expectClose(out, [0, 0]);
  });
});
