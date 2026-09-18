// Decision tree and tree ensemble execution for the reference engine.

import type { Tree, TreeEnsemble, TreeSplitOp } from '../ir.js';
import { scalarToNumber } from '../ir.js';
import { applyPostTransform, tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';
import { evalPredicate } from './predicates.js';
import type { ColumnResolver } from './predicates.js';

// ── Tree execution ────────────────────────────────────────────────────────────

export function executeTree(
  tree: Tree,
  flatInputs: Float64Array,  // shape [N * numSlots]
  numSlots: number,
  N: number,
  resolved?: ResolvedModel,
): Float64Array {
  const tensorIndex = resolved?.tensorIndex ?? new Map();
  const leafVecSize = tree.leaf_width ?? 0;
  const outputWidth = leafVecSize > 1 ? leafVecSize : 1;
  const out = new Float64Array(N * outputWidth);

  // Pre-extract tensor data arrays
  const splitThresholdTensor = resolveTensorValue(tree.split_threshold, tensorIndex);
  const splitThresholds = splitThresholdTensor ? tensorToData(splitThresholdTensor).data as Float64Array : null;
  // Detect if the split thresholds were originally stored as float32 (e.g. XGBoost models).
  // In that case, feature values must be cast to float32 before comparison to match native behaviour.
  const threshIsFloat32 = splitThresholdTensor != null &&
    !splitThresholdTensor.float64_data?.length &&
    !splitThresholdTensor.raw_data &&
    (splitThresholdTensor.float32_data?.length ?? 0) > 0;
  const leafValueTensor = resolveTensorValue(tree.leaf_value, tensorIndex);
  const leafValues = leafValueTensor ? tensorToData(leafValueTensor).data as Float64Array : null;
  const leafVectorTensor = resolveTensorValue(tree.leaf_vector, tensorIndex);
  const leafVector = leafVectorTensor ? tensorToData(leafVectorTensor).data as Float64Array : null;

  // Build complex predicate index for O(1) lookup
  const complexMap = new Map<number, import('../ir.js').Predicate>();
  for (const cp of tree.complex_predicates ?? []) {
    complexMap.set(cp.node_index, cp.predicate);
  }

  for (let row = 0; row < N; row++) {
    const rowOffset = row * numSlots;
    const _resolver: ColumnResolver = (_, __) => undefined; // unused for compact splits

    let nodeIdx = 0;
    let _loopGuard = 0;
    while (true) {
      if (++_loopGuard > 100_000) throw new Error(`Tree traversal loop at node ${nodeIdx} row ${row}`);
      const kind = tree.node_kind?.[nodeIdx] ?? 'LEAF';
      if (kind === 'LEAF') {
        if (outputWidth > 1) {
          const vecIdx = (tree.leaf_vector_index?.[nodeIdx] ?? 0);
          for (let k = 0; k < outputWidth; k++) {
            out[row * outputWidth + k] = leafVector?.[vecIdx + k] ?? 0;
          }
        } else {
          out[row] = leafValues?.[nodeIdx] ?? 0;
        }
        break;
      }

      // BRANCH: evaluate split
      const childOffset = tree.children_offset?.[nodeIdx] ?? 0;

      const complexPred = complexMap.get(nodeIdx);
      if (complexPred) {
        // Full predicate fallback — build resolver from flat slot space
        const slotResolver: ColumnResolver = (name, _r) => {
          const slotIdx = parseInt(name);
          if (!isNaN(slotIdx)) return flatInputs[rowOffset + slotIdx];
          return undefined;
        };
        const result = evalPredicate(complexPred, slotResolver, row);
        const goLeft = result !== false;
        nodeIdx = tree.children_index?.[childOffset + (goLeft ? 0 : 1)] ?? 0;
        continue;
      }

      const featureIdx = tree.split_feature?.[nodeIdx] ?? 0;
      const featureValRaw = flatInputs[rowOffset + featureIdx];
      const isMissing = isNaN(featureValRaw);

      if (isMissing) {
        nodeIdx = tree.default_child?.[nodeIdx] ?? 0;
        continue;
      }

      // When thresholds are float32 (e.g. XGBoost), cast feature value to float32
      // so that boundary comparisons match the native engine's behaviour.
      const featureVal = threshIsFloat32 ? Math.fround(featureValRaw) : featureValRaw;
      const op: TreeSplitOp = tree.split_op?.[nodeIdx] ?? 'SPLIT_OP_UNSPECIFIED';
      const threshold = splitThresholds?.[nodeIdx] ?? 0;
      const goLeft = evalSplitOp(op, featureVal, threshold, nodeIdx, tree);
      nodeIdx = tree.children_index?.[childOffset + (goLeft ? 0 : 1)] ?? 0;
    }
  }

  return out;
}

function evalSplitOp(
  op: TreeSplitOp,
  val: number,
  threshold: number,
  nodeIdx: number,
  tree: Tree,
): boolean {
  switch (op) {
    case 'LESS_THAN': return val < threshold;
    case 'LESS_OR_EQUAL': return val <= threshold;
    case 'GREATER_THAN': return val > threshold;
    case 'GREATER_OR_EQUAL': return val >= threshold;
    case 'EQUAL': return val === threshold;
    case 'NOT_EQUAL': return val !== threshold;
    case 'IS_MISSING': return isNaN(val);
    case 'IN_SET': {
      const offset = tree.category_set_offset?.[nodeIdx] ?? 0;
      const count = tree.category_set_count?.[nodeIdx] ?? 0;
      for (let j = offset; j < offset + count; j++) {
        if (tree.category_set?.[j] === val) return true;
      }
      return false;
    }
    case 'NOT_IN_SET': {
      const offset = tree.category_set_offset?.[nodeIdx] ?? 0;
      const count = tree.category_set_count?.[nodeIdx] ?? 0;
      for (let j = offset; j < offset + count; j++) {
        if (tree.category_set?.[j] === val) return false;
      }
      return true;
    }
    default: return true;
  }
}

// ── TreeEnsemble execution ────────────────────────────────────────────────────

export function executeTreeEnsemble(
  ensemble: TreeEnsemble,
  flatInputs: Float64Array,
  numSlots: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array {
  const trees = ensemble.trees ?? [];
  if (trees.length === 0) return new Float64Array(N);

  // Determine per-tree output width from first tree
  const firstTree = trees[0];
  const leafVecSize = firstTree.leaf_width ?? 0;
  const perTreeWidth = leafVecSize > 1 ? leafVecSize : 1;

  const treeOutputs: Float64Array[] = trees.map(t =>
    executeTree(t, flatInputs, numSlots, N, resolved),
  );

  const weightsTensor = resolveTensorValue(ensemble.tree_weights, resolved.tensorIndex);
  const weightsData = weightsTensor ? tensorToData(weightsTensor).data as Float64Array : null;
  const aggregation = ensemble.aggregation ?? 'SUM';
  const baseScore = ensemble.base_score != null ? scalarToNumber(ensemble.base_score) : null;
  const treeGroup = ensemble.tree_group;

  // Determine output width after aggregation
  const _numClasses = perTreeWidth > 1 ? perTreeWidth : 1;
  let outWidth: number;

  if (treeGroup && treeGroup.length === trees.length) {
    // multiclass with class-grouped trees
    outWidth = Math.max(...treeGroup) + 1;
  } else {
    outWidth = perTreeWidth;
  }

  const rawScores = new Float64Array(N * outWidth);

  if (baseScore !== null && baseScore !== undefined) {
    rawScores.fill(baseScore);
  }

  switch (aggregation) {
    case 'SUM':
    case 'WEIGHTED_SUM': {
      for (let t = 0; t < trees.length; t++) {
        const w = weightsData?.[t] ?? 1;
        const treeOut = treeOutputs[t];
        if (treeGroup) {
          const classIdx = treeGroup[t];
          for (let row = 0; row < N; row++) {
            rawScores[row * outWidth + classIdx] += w * treeOut[row];
          }
        } else if (perTreeWidth > 1) {
          for (let row = 0; row < N; row++) {
            for (let k = 0; k < perTreeWidth; k++) {
              rawScores[row * outWidth + k] += w * treeOut[row * perTreeWidth + k];
            }
          }
        } else {
          for (let row = 0; row < N; row++) {
            rawScores[row] += w * treeOut[row];
          }
        }
      }
      break;
    }

    case 'AVERAGE':
    case 'WEIGHTED_AVERAGE': {
      let totalWeight = 0;
      for (let t = 0; t < trees.length; t++) {
        const w = weightsData?.[t] ?? 1;
        totalWeight += w;
        const treeOut = treeOutputs[t];
        for (let row = 0; row < N; row++) {
          if (perTreeWidth > 1) {
            for (let k = 0; k < perTreeWidth; k++) {
              rawScores[row * outWidth + k] += w * treeOut[row * perTreeWidth + k];
            }
          } else {
            rawScores[row] += w * treeOut[row];
          }
        }
      }
      if (totalWeight > 0) {
        for (let i = 0; i < rawScores.length; i++) rawScores[i] /= totalWeight;
      }
      break;
    }

    case 'SOFT_VOTE': {
      // Average per-class tree outputs. With treeGroup, each tree contributes to one class.
      const classCounts = new Float64Array(outWidth);
      for (let t = 0; t < trees.length; t++) {
        const w = weightsData?.[t] ?? 1;
        const treeOut = treeOutputs[t];
        if (treeGroup) {
          const classIdx = treeGroup[t];
          classCounts[classIdx] += w;
          for (let row = 0; row < N; row++) {
            rawScores[row * outWidth + classIdx] += w * treeOut[row];
          }
        } else {
          for (let row = 0; row < N; row++) {
            for (let k = 0; k < outWidth; k++) {
              rawScores[row * outWidth + k] += w * (treeOut[row * outWidth + k] ?? treeOut[row]);
            }
          }
          classCounts[0] += w;
        }
      }
      for (let k = 0; k < outWidth; k++) {
        if (classCounts[k] > 0) {
          for (let row = 0; row < N; row++) rawScores[row * outWidth + k] /= classCounts[k];
        }
      }
      break;
    }

    case 'MAJORITY_VOTE': {
      const votes = new Array(N).fill(0).map(() => new Map<number, number>());
      for (let t = 0; t < trees.length; t++) {
        const treeOut = treeOutputs[t];
        for (let row = 0; row < N; row++) {
          const label = Math.round(treeOut[row]);
          votes[row].set(label, (votes[row].get(label) ?? 0) + 1);
        }
      }
      for (let row = 0; row < N; row++) {
        let bestLabel = 0, bestCount = 0;
        for (const [label, count] of votes[row]) {
          if (count > bestCount) { bestCount = count; bestLabel = label; }
        }
        rawScores[row] = bestLabel;
      }
      break;
    }

    case 'MIN': {
      rawScores.fill(Infinity);
      for (const treeOut of treeOutputs) {
        for (let i = 0; i < rawScores.length; i++) {
          rawScores[i] = Math.min(rawScores[i], treeOut[i]);
        }
      }
      break;
    }

    case 'MAX': {
      rawScores.fill(-Infinity);
      for (const treeOut of treeOutputs) {
        for (let i = 0; i < rawScores.length; i++) {
          rawScores[i] = Math.max(rawScores[i], treeOut[i]);
        }
      }
      break;
    }

    default:
      // SUM fallback
      for (const treeOut of treeOutputs) {
        for (let i = 0; i < rawScores.length; i++) rawScores[i] += treeOut[i];
      }
  }

  return applyPostTransform(rawScores, ensemble.post_transform, outWidth);
}
