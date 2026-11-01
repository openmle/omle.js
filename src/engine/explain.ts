// Structured model explain adapters.
//
// Rerun the model body for row 0 of the batch, capturing the intermediate
// values needed by the per-node deep inspection UI.

import type {
  Node, Tree, TreeEnsemble, TreeSplitOp, Linear, NaiveBayes, Clustering,
  PrototypeClustering, GaussianMixtureClustering,
} from '../ir.js';
import { scalarToNumber } from '../ir.js';
import type { TensorData } from './ops.js';
import { tensorToData, applyPostTransform, matmulBias, tensorCols } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue, expandNodeInputs } from '../resolve.js';

// ── Public types ──────────────────────────────────────────────────────────────

export interface TreePathStep {
  nodeIndex:    number;
  featureIndex: number;
  featureName:  string;
  op:           string;
  threshold:    number;
  value:        number;
  goLeft:       boolean;
}

export interface TreeExplainItem {
  treeIndex:  number;
  weight:     number;
  path:       TreePathStep[];
  leafValue:  number[];
}

export interface TreeEnsembleExplain {
  type:        'tree_ensemble';
  aggregation: string;
  baseScore:   number | null;
  trees:       TreeExplainItem[];
  classScores: number[];   // after post-transform
}

export interface LinearContribution {
  featureIndex:   number;
  name:           string;
  value:          number;
  coefficients:   number[];  // one per output class
  contributions:  number[];  // value × coefficient, one per class
  absContribution: number;   // sum of |contributions| — used for sort
}

export interface LinearExplain {
  type:         'linear';
  intercept:    number[];
  contributions: LinearContribution[];  // sorted by absContribution desc
  rawScores:    number[];
  finalScores:  number[];
}

export interface NaiveBayesFeatureContrib {
  featureIndex:   number;
  name:           string;
  value:          number;
  logLikelihoods: number[];  // one per class
}

export interface NaiveBayesExplain {
  type:               'naive_bayes';
  variant:            'gaussian' | 'multinomial' | 'bernoulli' | 'categorical';
  numClasses:         number;
  logPriors:          number[];
  featureContributions: NaiveBayesFeatureContrib[];
  classLogScores:     number[];  // logPrior + Σ logLikelihoods per class
}

export interface CentroidDistance {
  centroidIndex: number;
  distance:      number;
}

export interface ClusteringExplain {
  type:              'clustering';
  variant:           'prototype' | 'gmm';
  distanceMeasure:   string;
  distances:         CentroidDistance[];  // all centroids, sorted by distance
  selectedCentroid:  number;
}

export type ModelExplain =
  | TreeEnsembleExplain
  | LinearExplain
  | NaiveBayesExplain
  | ClusteringExplain;

// ── Entry point ───────────────────────────────────────────────────────────────

// Called in runWithSteps() after executing a node.
// Returns undefined for node types without a structured explain (NN, SVM, composite, etc.).
export function computeExplain(
  node: Node,
  namespace: Map<string, TensorData>,
  resolved: ResolvedModel,
): ModelExplain | undefined {
  try {
    const inputNames = expandNodeInputs(node.inputs ?? []);
    if (node.tree_ensemble) return explainTreeEnsemble(node.tree_ensemble, inputNames, namespace, resolved);
    if (node.linear)        return explainLinear(node.linear, inputNames, namespace, resolved);
    if (node.naive_bayes)   return explainNaiveBayes(node.naive_bayes, inputNames, namespace, resolved);
    if (node.clustering)    return explainClustering(node.clustering, inputNames, namespace, resolved);
    return undefined;
  } catch {
    return undefined;
  }
}

// ── Feature extraction ────────────────────────────────────────────────────────

interface FeatureSet {
  featureNames: string[];
  flatRow:      Float64Array;
  numSlots:     number;
}

function extractFeatures(
  inputNames: string[],
  namespace: Map<string, TensorData>,
): FeatureSet {
  const featureNames: string[] = [];
  let numSlots = 0;

  for (const name of inputNames) {
    const td = namespace.get(name);
    if (!td) continue;
    const cols = tensorCols(td);
    if (cols === 1) {
      featureNames.push(name);
    } else {
      for (let c = 0; c < cols; c++) featureNames.push(`${name}[${c}]`);
    }
    numSlots += cols;
  }

  const flatRow = new Float64Array(numSlots);
  let offset = 0;
  for (const name of inputNames) {
    const td = namespace.get(name);
    if (!td) continue;
    const data = td.data as Float64Array;
    const cols = tensorCols(td);
    for (let c = 0; c < cols; c++) flatRow[offset + c] = data[c];  // row 0
    offset += cols;
  }

  return { featureNames, flatRow, numSlots };
}

// ── Tree ensemble explain ─────────────────────────────────────────────────────

function explainTreeEnsemble(
  ensemble: TreeEnsemble,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  resolved: ResolvedModel,
): TreeEnsembleExplain {
  const { featureNames, flatRow, numSlots } = extractFeatures(inputNames, namespace);
  const trees = ensemble.trees ?? [];

  const weightsTensor = resolveTensorValue(ensemble.tree_weights, resolved.tensorIndex);
  const weightsData = weightsTensor ? tensorToData(weightsTensor).data as Float64Array : null;
  const treeItems: TreeExplainItem[] = trees.map((tree, i) => {
    const { path, leafValue } = traverseTree(tree, flatRow, numSlots, featureNames, resolved);
    return { treeIndex: i, weight: weightsData?.[i] ?? 1, path, leafValue };
  });

  // Aggregate for row 0
  const firstTree = trees[0];
  const leafVecSize = firstTree?.leaf_width ?? 0;
  const perTreeWidth = leafVecSize > 1 ? leafVecSize : 1;
  const treeGroup = ensemble.tree_group;
  const outWidth = (treeGroup && treeGroup.length === trees.length)
    ? Math.max(...treeGroup) + 1
    : perTreeWidth;

  const rawScores = new Float64Array(outWidth);
  const explainBase = resolveTensorValue(ensemble.base_scores, resolved.tensorIndex);
  if (explainBase) {
    const bs = Array.from(tensorToData(explainBase).data as ArrayLike<number>).map(Number);
    if (bs.length > 0) rawScores.fill(bs[0]);
  }

  const agg = ensemble.aggregation ?? 'SUM';
  let totalWeight = 0;

  for (const item of treeItems) {
    const w = item.weight;
    totalWeight += w;
    if (treeGroup) {
      rawScores[treeGroup[item.treeIndex]] += w * (item.leafValue[0] ?? 0);
    } else if (perTreeWidth > 1) {
      for (let k = 0; k < perTreeWidth; k++) rawScores[k] += w * (item.leafValue[k] ?? 0);
    } else {
      rawScores[0] += w * (item.leafValue[0] ?? 0);
    }
  }

  if ((agg === 'AVERAGE' || agg === 'WEIGHTED_AVERAGE') && totalWeight > 0) {
    for (let i = 0; i < rawScores.length; i++) rawScores[i] /= totalWeight;
  }

  const finalScores = applyPostTransform(rawScores, ensemble.post_transform, outWidth);
  return {
    type: 'tree_ensemble',
    aggregation: agg,
    baseScore: explainBase
      ? Number((tensorToData(explainBase).data as ArrayLike<number>)[0] ?? 0)
      : null,
    trees: treeItems,
    classScores: Array.from(finalScores),
  };
}

function traverseTree(
  tree: Tree,
  flatRow: Float64Array,
  numSlots: number,
  featureNames: string[],
  resolved: ResolvedModel,
): { path: TreePathStep[]; leafValue: number[] } {
  const path: TreePathStep[] = [];
  const leafVecSize = tree.leaf_width ?? 0;
  const outWidth = leafVecSize > 1 ? leafVecSize : 1;

  // Pre-extract tensor data
  const splitThresholdTensor = resolveTensorValue(tree.split_threshold, resolved.tensorIndex);
  const splitThresholds = splitThresholdTensor ? tensorToData(splitThresholdTensor).data as Float64Array : null;
  const leafValueTensor = resolveTensorValue(tree.leaf_value, resolved.tensorIndex);
  const leafValues = leafValueTensor ? tensorToData(leafValueTensor).data as Float64Array : null;
  const leafVectorTensor = resolveTensorValue(tree.leaf_vector, resolved.tensorIndex);
  const leafVectorData = leafVectorTensor ? tensorToData(leafVectorTensor).data as Float64Array : null;

  let nodeIdx = 0;

  for (let iter = 0; iter < 100_000; iter++) {
    const kind = tree.node_kind?.[nodeIdx] ?? 'LEAF';
    if (kind === 'LEAF') {
      let leafValue: number[];
      if (outWidth > 1) {
        const vecIdx = tree.leaf_vector_index?.[nodeIdx] ?? 0;
        leafValue = Array.from(leafVectorData?.slice(vecIdx, vecIdx + outWidth) ?? []);
      } else {
        leafValue = [leafValues?.[nodeIdx] ?? 0];
      }
      return { path, leafValue };
    }

    const featureIdx   = tree.split_feature?.[nodeIdx] ?? 0;
    const featureName  = featureIdx < featureNames.length ? featureNames[featureIdx] : `slot_${featureIdx}`;
    const featureVal   = featureIdx < numSlots ? flatRow[featureIdx] : NaN;
    const isMissing    = isNaN(featureVal);
    const op           = (tree.split_op?.[nodeIdx] ?? 'LESS_THAN') as TreeSplitOp;
    const threshold    = splitThresholds?.[nodeIdx] ?? 0;
    const defChild     = tree.default_child?.[nodeIdx] ?? 0;
    const goLeft       = isMissing ? defChild === 0 : evalSplit(op, featureVal, threshold, nodeIdx, tree);

    path.push({ nodeIndex: nodeIdx, featureIndex: featureIdx, featureName, op, threshold, value: featureVal, goLeft });

    const childOffset = tree.children_offset?.[nodeIdx] ?? 0;
    nodeIdx = tree.children_index?.[childOffset + (isMissing ? defChild : (goLeft ? 0 : 1))] ?? 0;
  }
  return { path: [], leafValue: [0] };
}

function evalSplit(op: TreeSplitOp, val: number, threshold: number, nodeIdx: number, tree: Tree): boolean {
  switch (op) {
    case 'LESS_THAN':          return val < threshold;
    case 'LESS_OR_EQUAL':      return val <= threshold;
    case 'GREATER_THAN':       return val > threshold;
    case 'GREATER_OR_EQUAL':   return val >= threshold;
    case 'EQUAL':              return val === threshold;
    case 'NOT_EQUAL':          return val !== threshold;
    case 'IS_MISSING':         return isNaN(val);
    case 'IN_SET': {
      const off = tree.category_set_offset?.[nodeIdx] ?? 0;
      const cnt = tree.category_set_count?.[nodeIdx] ?? 0;
      for (let j = off; j < off + cnt; j++) if (tree.category_set?.[j] === val) return true;
      return false;
    }
    case 'NOT_IN_SET': {
      const off = tree.category_set_offset?.[nodeIdx] ?? 0;
      const cnt = tree.category_set_count?.[nodeIdx] ?? 0;
      for (let j = off; j < off + cnt; j++) if (tree.category_set?.[j] === val) return false;
      return true;
    }
    default: return true;
  }
}

// ── Linear explain ────────────────────────────────────────────────────────────

function explainLinear(
  linear: Linear,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  resolved: ResolvedModel,
): LinearExplain {
  const { featureNames, flatRow, numSlots } = extractFeatures(inputNames, namespace);
  const { tensorIndex } = resolved;

  const coeffTensor = resolveTensorValue(linear.coefficients, tensorIndex);
  if (!coeffTensor) throw new Error('Linear explain: coefficients tensor is missing');
  const coeffArr    = tensorToData(coeffTensor).data as Float64Array;
  const coeffShape  = coeffTensor.type?.shape ?? [];
  const outWidth    = coeffShape.length === 2 ? (coeffShape[0] ?? 1) : 1;

  let interceptArr: Float64Array = new Float64Array(outWidth);
  if (linear.intercept) {
    const interceptTensor = resolveTensorValue(linear.intercept, tensorIndex);
    if (interceptTensor) interceptArr = tensorToData(interceptTensor).data as Float64Array<ArrayBuffer>;
  }

  const contributions: LinearContribution[] = [];
  for (let i = 0; i < numSlots; i++) {
    const val = flatRow[i];
    const coefficients: number[] = [];
    const contribs: number[] = [];
    for (let k = 0; k < outWidth; k++) {
      const coeff = outWidth === 1 ? (coeffArr[i] ?? 0) : (coeffArr[k * numSlots + i] ?? 0);
      coefficients.push(coeff);
      contribs.push(val * coeff);
    }
    contributions.push({
      featureIndex:   i,
      name:           featureNames[i] ?? `slot_${i}`,
      value:          val,
      coefficients,
      contributions:  contribs,
      absContribution: contribs.reduce((s, c) => s + Math.abs(c), 0),
    });
  }
  contributions.sort((a, b) => b.absContribution - a.absContribution);

  const rawScores   = matmulBias(flatRow, 1, numSlots, coeffArr, outWidth, interceptArr);
  const finalScores = applyPostTransform(rawScores, linear.post_transform, outWidth);

  return {
    type: 'linear',
    intercept:     Array.from(interceptArr),
    contributions,
    rawScores:     Array.from(rawScores),
    finalScores:   Array.from(finalScores),
  };
}

// ── Naive Bayes explain ───────────────────────────────────────────────────────

function explainNaiveBayes(
  nb: NaiveBayes,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  resolved: ResolvedModel,
): NaiveBayesExplain {
  const { featureNames, flatRow, numSlots } = extractFeatures(inputNames, namespace);
  const { tensorIndex } = resolved;

  const priorsTensor = resolveTensorValue(nb.class_log_priors, tensorIndex);
  if (!priorsTensor) throw new Error('NaiveBayes explain: class_log_priors tensor is missing');
  const logPriors  = Array.from(tensorToData(priorsTensor).data as Float64Array);
  const numClasses = logPriors.length;
  const classLogScores = [...logPriors];
  const featureContributions: NaiveBayesFeatureContrib[] = [];

  const pushFeatures = (getLLs: (f: number) => number[]) => {
    for (let f = 0; f < numSlots; f++) {
      const lls = getLLs(f);
      for (let c = 0; c < numClasses; c++) classLogScores[c] += lls[c];
      featureContributions.push({ featureIndex: f, name: featureNames[f] ?? `slot_${f}`, value: flatRow[f], logLikelihoods: lls });
    }
  };

  if (nb.gaussian) {
    const meansTensor = resolveTensorValue(nb.gaussian.means, tensorIndex);
    const varsTensor = resolveTensorValue(nb.gaussian.variances, tensorIndex);
    if (!meansTensor) throw new Error('GaussianNaiveBayes explain: means tensor is missing');
    if (!varsTensor) throw new Error('GaussianNaiveBayes explain: variances tensor is missing');
    const means = tensorToData(meansTensor).data as Float64Array;
    const vars  = tensorToData(varsTensor).data as Float64Array;
    const eps   = nb.gaussian.variance_epsilon != null ? scalarToNumber(nb.gaussian.variance_epsilon) : 0;
    pushFeatures(f => Array.from({ length: numClasses }, (_, c) => {
      const x = flatRow[f], mu = means[c * numSlots + f], v = vars[c * numSlots + f] + eps;
      return -0.5 * (Math.log(2 * Math.PI * v) + (x - mu) ** 2 / v);
    }));
    return { type: 'naive_bayes', variant: 'gaussian', numClasses, logPriors, featureContributions, classLogScores };
  }

  if (nb.multinomial) {
    const flpTensor = resolveTensorValue(nb.multinomial.feature_log_prob, tensorIndex);
    if (!flpTensor) throw new Error('MultinomialNaiveBayes explain: feature_log_prob tensor is missing');
    const flp = tensorToData(flpTensor).data as Float64Array;
    pushFeatures(f => Array.from({ length: numClasses }, (_, c) => flatRow[f] * flp[c * numSlots + f]));
    return { type: 'naive_bayes', variant: 'multinomial', numClasses, logPriors, featureContributions, classLogScores };
  }

  if (nb.bernoulli) {
    const flpTensor2 = resolveTensorValue(nb.bernoulli.feature_log_prob, tensorIndex);
    if (!flpTensor2) throw new Error('BernoulliNaiveBayes explain: feature_log_prob tensor is missing');
    const flp       = tensorToData(flpTensor2).data as Float64Array;
    const binThresh = nb.bernoulli.binarize_threshold != null ? scalarToNumber(nb.bernoulli.binarize_threshold) : null;
    const logNeg    = Float64Array.from(flp, v => Math.log(1 - Math.exp(v) + 1e-10));
    pushFeatures(f => {
      const raw = flatRow[f];
      const xi  = binThresh !== null ? (raw > binThresh ? 1 : 0) : raw;
      return Array.from({ length: numClasses }, (_, c) =>
        xi * flp[c * numSlots + f] + (1 - xi) * logNeg[c * numSlots + f]);
    });
    return { type: 'naive_bayes', variant: 'bernoulli', numClasses, logPriors, featureContributions, classLogScores };
  }

  if (nb.categorical) {
    const clpTensor = resolveTensorValue(nb.categorical.category_log_prob, tensorIndex);
    if (!clpTensor) throw new Error('CategoricalNaiveBayes explain: category_log_prob tensor is missing');
    const clp       = tensorToData(clpTensor).data as Float64Array;
    const catOffset = nb.categorical.category_offset ?? [];
    const catCount  = nb.categorical.category_count ?? [];
    const rowSize   = catOffset.length > 0
      ? (catOffset[catOffset.length - 1] + (catCount[catCount.length - 1] ?? 0))
      : numSlots;
    pushFeatures(f => {
      const catIdx  = Math.round(flatRow[f]);
      const offset  = catOffset[f] ?? f;
      const count   = catCount[f] ?? 1;
      const safeIdx = Math.max(0, Math.min(catIdx, count - 1));
      return Array.from({ length: numClasses }, (_, c) => clp[c * rowSize + offset + safeIdx]);
    });
    return { type: 'naive_bayes', variant: 'categorical', numClasses, logPriors, featureContributions, classLogScores };
  }

  return { type: 'naive_bayes', variant: 'gaussian', numClasses, logPriors, featureContributions: [], classLogScores };
}

// ── Clustering explain ────────────────────────────────────────────────────────

function explainClustering(
  clustering: Clustering,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  resolved: ResolvedModel,
): ClusteringExplain {
  const { flatRow, numSlots } = extractFeatures(inputNames, namespace);
  if (clustering.prototype)       return explainPrototype(clustering.prototype, flatRow, numSlots, resolved);
  if (clustering.gaussian_mixture) return explainGMM(clustering.gaussian_mixture, flatRow, numSlots, resolved);
  return { type: 'clustering', variant: 'prototype', distanceMeasure: 'EUCLIDEAN', distances: [], selectedCentroid: 0 };
}

function explainPrototype(
  proto: PrototypeClustering,
  flatRow: Float64Array,
  numSlots: number,
  resolved: ResolvedModel,
): ClusteringExplain {
  const centersTensor = resolveTensorValue(proto.centers, resolved.tensorIndex);
  if (!centersTensor) throw new Error('PrototypeClustering explain: centers tensor is missing');
  const centers       = tensorToData(centersTensor).data as Float64Array;
  const numClusters   = centersTensor.type?.shape?.[0] ?? 1;
  const measure       = proto.distance_measure ?? 'EUCLIDEAN';

  let selectedCentroid = 0;
  let minDist = Infinity;
  const distances: CentroidDistance[] = [];

  for (let c = 0; c < numClusters; c++) {
    const dist = centroidDist(flatRow, 0, numSlots, centers, c * numSlots, measure);
    distances.push({ centroidIndex: c, distance: dist });
    if (dist < minDist) { minDist = dist; selectedCentroid = c; }
  }

  return { type: 'clustering', variant: 'prototype', distanceMeasure: measure, distances, selectedCentroid };
}

function explainGMM(
  gmm: GaussianMixtureClustering,
  flatRow: Float64Array,
  numSlots: number,
  resolved: ResolvedModel,
): ClusteringExplain {
  const gmmWeightsTensor = resolveTensorValue(gmm.weights, resolved.tensorIndex);
  const gmmMeansTensor   = resolveTensorValue(gmm.means, resolved.tensorIndex);
  if (!gmmWeightsTensor) throw new Error('GMM explain: weights tensor is missing');
  if (!gmmMeansTensor) throw new Error('GMM explain: means tensor is missing');
  const weights      = tensorToData(gmmWeightsTensor).data as Float64Array;
  const means        = tensorToData(gmmMeansTensor).data as Float64Array;
  const numClusters  = weights.length;

  let selectedCentroid = 0;
  let bestLogProb = -Infinity;
  const distances: CentroidDistance[] = [];

  for (let c = 0; c < numClusters; c++) {
    let sq = 0;
    for (let f = 0; f < numSlots; f++) { const d = flatRow[f] - means[c * numSlots + f]; sq += d * d; }
    const logProb = Math.log(weights[c] + 1e-10) - 0.5 * sq;
    distances.push({ centroidIndex: c, distance: Math.sqrt(sq) });
    if (logProb > bestLogProb) { bestLogProb = logProb; selectedCentroid = c; }
  }

  return { type: 'clustering', variant: 'gmm', distanceMeasure: 'EUCLIDEAN', distances, selectedCentroid };
}

function centroidDist(
  a: Float64Array, aOff: number, n: number,
  b: Float64Array, bOff: number,
  measure: string,
): number {
  switch (measure) {
    case 'EUCLIDEAN': {
      let s = 0; for (let i = 0; i < n; i++) { const d = a[aOff+i]-b[bOff+i]; s += d*d; } return Math.sqrt(s);
    }
    case 'SQUARED_EUCLIDEAN': {
      let s = 0; for (let i = 0; i < n; i++) { const d = a[aOff+i]-b[bOff+i]; s += d*d; } return s;
    }
    case 'MANHATTAN': {
      let s = 0; for (let i = 0; i < n; i++) s += Math.abs(a[aOff+i]-b[bOff+i]); return s;
    }
    case 'COSINE': {
      let dot = 0, nA = 0, nB = 0;
      for (let i = 0; i < n; i++) { dot += a[aOff+i]*b[bOff+i]; nA += a[aOff+i]**2; nB += b[bOff+i]**2; }
      const den = Math.sqrt(nA) * Math.sqrt(nB);
      return den > 0 ? 1 - dot/den : 1;
    }
    default: {
      let s = 0; for (let i = 0; i < n; i++) { const d = a[aOff+i]-b[bOff+i]; s += d*d; } return Math.sqrt(s);
    }
  }
}
