// Anomaly detection execution for the reference engine.
//
// The score returned here is the model's raw score_samples, matching the
// SCORE output role: the detector's own decision function, before the
// `offset` that would centre it into a decision value. Converters record the
// polarity in raw_score_polarity; this engine reproduces the value, not an
// interpretation of it.

import type { AnomalyDetection, Tree } from '../ir.js';
import { scalarToNumber } from '../ir.js';
import { tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';

function tensorNumbers(
  tv: unknown,
  resolved: ResolvedModel,
): Float64Array | null {
  const t = resolveTensorValue(tv as never, resolved.tensorIndex);
  if (!t) return null;
  const d = tensorToData(t).data as ArrayLike<number>;
  const out = new Float64Array(d.length);
  for (let i = 0; i < d.length; i++) out[i] = Number(d[i]);
  return out;
}

function kernel(
  x: Float64Array, xOff: number,
  sv: Float64Array, svOff: number,
  n: number, kind: string,
  gamma: number, degree: number, coef0: number,
): number {
  switch (kind) {
    case 'LINEAR': {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += x[xOff + i] * sv[svOff + i];
      return dot;
    }
    case 'POLY': {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += x[xOff + i] * sv[svOff + i];
      return Math.pow(gamma * dot + coef0, degree);
    }
    case 'SIGMOID': {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += x[xOff + i] * sv[svOff + i];
      return Math.tanh(gamma * dot + coef0);
    }
    case 'RBF':
    default: {
      let d2 = 0;
      for (let i = 0; i < n; i++) {
        const d = x[xOff + i] - sv[svOff + i];
        d2 += d * d;
      }
      return Math.exp(-gamma * d2);
    }
  }
}


// sklearn's _average_path_length: the expected path length of an unsuccessful
// BST search over n points, used to normalise isolation depths.
const EULER_GAMMA = 0.5772156649015329;  // nearest double to γ

function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + EULER_GAMMA) - 2 * (n - 1) / n;
}

function minkowski(
  a: Float64Array, aOff: number,
  b: Float64Array, bOff: number,
  d: number, pp: number,
): number {
  if (pp === 2) {
    let s = 0;
    for (let i = 0; i < d; i++) { const t = a[aOff + i] - b[bOff + i]; s += t * t; }
    return Math.sqrt(s);
  }
  if (pp === 1) {
    let s = 0;
    for (let i = 0; i < d; i++) s += Math.abs(a[aOff + i] - b[bOff + i]);
    return s;
  }
  if (!Number.isFinite(pp)) {
    let m = 0;
    for (let i = 0; i < d; i++) m = Math.max(m, Math.abs(a[aOff + i] - b[bOff + i]));
    return m;
  }
  let s = 0;
  for (let i = 0; i < d; i++) s += Math.pow(Math.abs(a[aOff + i] - b[bOff + i]), pp);
  return Math.pow(s, 1 / pp);
}

/**
 * Walk one isolation tree and return its stored leaf value.
 *
 * Isolation trees are scored by path length, not by the ordinary tree
 * semantics: the converter folds `depth + c(n_node_samples)` into the leaf, so
 * the walk just has to reach the right leaf. Branching is `value <= threshold`
 * goes left, matching sklearn's tree layout.
 */
function isolationLeaf(
  t: Tree, thresholds: Float64Array, leaves: Float64Array,
  x: Float64Array, xOff: number, numFeatures: number,
): number {
  const kind = t.node_kind ?? [];
  const feat = t.split_feature ?? [];
  const childOffset = t.children_offset ?? [];
  const childIndex = t.children_index ?? [];
  const childCount = t.children_count ?? [];
  let node = 0;
  let guard = 0;
  while (node >= 0 && node < kind.length && kind[node] !== 'LEAF') {
    if (++guard > kind.length) break;          // malformed tree: stop, don't spin
    const f = feat[node] ?? -1;
    const v = (f >= 0 && f < numFeatures) ? x[xOff + f] : 0;
    const off = childOffset[node] ?? -1;
    if ((childCount[node] ?? 0) < 2 || off < 0 || off + 1 >= childIndex.length) break;
    node = v <= thresholds[node] ? childIndex[off] : childIndex[off + 1];
  }
  return (node >= 0 && node < leaves.length) ? leaves[node] : 0;
}

/**
 * Score one batch. Returns a [N] column, or null when the model uses an
 * implementation this engine does not have — the caller then leaves the
 * output unset rather than publishing a wrong score.
 */
export function executeAnomalyDetection(
  ad: AnomalyDetection,
  flatInputs: Float64Array,   // [N * numFeatures]
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array | null {
  // ── Linear one-class SVM: intercept + w · x ───────────────────────────────
  if (ad.linear_one_class_svm) {
    const lin = ad.linear_one_class_svm;
    const coef = tensorNumbers(lin.coefficients, resolved);
    if (!coef) return null;
    // intercept is a TensorValue here (a [1] tensor), not a Scalar.
    const interceptArr = tensorNumbers(lin.intercept, resolved);
    const intercept = interceptArr && interceptArr.length > 0 ? interceptArr[0] : 0;
    const out = new Float64Array(N);
    const nf = Math.min(numFeatures, coef.length);
    for (let row = 0; row < N; row++) {
      let acc = intercept;
      for (let i = 0; i < nf; i++) acc += coef[i] * flatInputs[row * numFeatures + i];
      out[row] = acc;
    }
    return out;
  }

  // ── Kernel one-class SVM: intercept + Σ dual_coef[s] · K(sv[s], x) ────────
  if (ad.one_class_svm) {
    const k = ad.one_class_svm.kernel_svm;
    if (!k) return null;
    const sv = tensorNumbers(k.support_vectors, resolved);
    const dual = tensorNumbers(k.dual_coefficients, resolved);
    if (!sv || !dual) return null;
    const kInterceptArr = tensorNumbers(k.intercept, resolved);
    const intercept = kInterceptArr && kInterceptArr.length > 0 ? kInterceptArr[0] : 0;
    const kind = k.kernel_type ?? 'RBF';
    // sklearn's default gamma is 'scale'; the converter resolves it to a
    // number, so a missing gamma here means the model never set one.
    const gamma = k.gamma != null ? scalarToNumber(k.gamma) : 1 / Math.max(numFeatures, 1);
    const degree = k.degree ?? 3;
    const coef0 = k.coef0 != null ? scalarToNumber(k.coef0) : 0;
    const nSV = dual.length;
    if (nSV === 0) return null;
    // support_vectors is [n_sv, n_features]; trust its own width rather than
    // the slot count, which may include columns the model does not use.
    const svWidth = Math.floor(sv.length / nSV) || numFeatures;
    const nf = Math.min(numFeatures, svWidth);
    const out = new Float64Array(N);
    for (let row = 0; row < N; row++) {
      let acc = intercept;
      for (let s = 0; s < nSV; s++) {
        acc += dual[s] * kernel(
          flatInputs, row * numFeatures, sv, s * svWidth, nf,
          kind, gamma, degree, coef0,
        );
      }
      out[row] = acc;
    }
    return out;
  }

  // ── Isolation forest: -2^(-mean_path / c(max_samples)) ───────────────────
  if (ad.isolation_forest) {
    const forest = ad.isolation_forest;
    const trees = forest.trees ?? [];
    if (trees.length === 0) return null;
    const norm = averagePathLength(Number(forest.max_samples ?? 0));
    if (!(norm > 0)) return null;

    // Resolve each tree's threshold and leaf tensors once, not per row.
    const prepared = trees.map(t => ({
      t,
      thr: tensorNumbers(t.split_threshold, resolved) ?? new Float64Array(0),
      leaf: tensorNumbers(t.leaf_value, resolved) ?? new Float64Array(0),
    }));

    const out = new Float64Array(N);
    for (let row = 0; row < N; row++) {
      let total = 0;
      for (const pt of prepared)
        total += isolationLeaf(pt.t, pt.thr, pt.leaf, flatInputs, row * numFeatures, numFeatures);
      out[row] = -Math.pow(2, -(total / prepared.length) / norm);
    }
    return out;
  }

  // ── Elliptic envelope: -mahalanobis(x) = -(x-loc)^T P (x-loc) ────────────
  if (ad.elliptic_envelope) {
    const env = ad.elliptic_envelope;
    const loc = tensorNumbers(env.location, resolved);
    const prec = tensorNumbers(env.precision, resolved);
    if (!loc || !prec) return null;
    const d = Math.min(numFeatures, loc.length);
    const out = new Float64Array(N);
    const diff = new Float64Array(d);
    for (let row = 0; row < N; row++) {
      for (let i = 0; i < d; i++) diff[i] = flatInputs[row * numFeatures + i] - loc[i];
      let acc = 0;
      for (let i = 0; i < d; i++) {
        let r = 0;
        for (let j = 0; j < d; j++) r += prec[i * d + j] * diff[j];
        acc += diff[i] * r;
      }
      out[row] = -acc;
    }
    return out;
  }

  // ── Local outlier factor: -(mean lrd of neighbours / lrd(x)) ─────────────
  if (ad.local_outlier_factor) {
    const lof = ad.local_outlier_factor;
    const ref = tensorNumbers(lof.reference_samples, resolved);
    if (!ref) return null;
    const d = numFeatures;
    const nRef = Math.floor(ref.length / Math.max(d, 1));
    if (nRef === 0) return null;
    const k = Math.max(1, Number(lof.n_neighbors ?? 20));
    // metric_params carries sklearn's effective_metric_params_ as strings;
    // only the Minkowski power changes the distance here.
    const pRaw = lof.metric_params?.['p'];
    const metric = lof.metric ?? 'minkowski';
    const pp = metric === 'manhattan' || metric === 'cityblock' ? 1
             : metric === 'chebyshev' ? Infinity
             : (pRaw != null && pRaw !== '' && !Number.isNaN(Number(pRaw)) ? Number(pRaw) : 2);

    /** k nearest reference points to `x`, excluding index `skip`. */
    function knn(x: Float64Array, xOff: number, skip: number):
        { idx: number[]; dist: number[] } {
      const all: { d: number; i: number }[] = [];
      for (let i = 0; i < nRef; i++) {
        if (i === skip) continue;
        all.push({ d: minkowski(x, xOff, ref!, i * d, d, pp), i });
      }
      // Ties must resolve the same way every call, so order by index when the
      // distances are equal -- otherwise a duplicated reference point can pick
      // a different neighbour in pass 1 than in pass 2.
      all.sort((a, b) => (a.d - b.d) || (a.i - b.i));
      const kk = Math.min(k, all.length);
      return { idx: all.slice(0, kk).map(e => e.i), dist: all.slice(0, kk).map(e => e.d) };
    }

    // Pass 1: k-distance of every reference point (itself excluded).
    const kdist = new Float64Array(nRef);
    for (let i = 0; i < nRef; i++) {
      const { dist } = knn(ref, i * d, i);
      kdist[i] = dist.length > 0 ? dist[dist.length - 1] : 0;
    }

    // lrd(x) = 1 / mean(max(k_distance(o), d(x, o)) for o in kNN(x)).
    // sklearn adds 1e-10 to the mean so duplicate points do not divide by zero.
    function lrd(x: Float64Array, xOff: number, skip: number): number {
      const { idx, dist } = knn(x, xOff, skip);
      if (idx.length === 0) return 0;
      let sum = 0;
      for (let j = 0; j < idx.length; j++) sum += Math.max(kdist[idx[j]], dist[j]);
      return 1 / (sum / idx.length + 1e-10);
    }

    // Pass 2: lrd of every reference point, which needs the k-distances above.
    const refLrd = new Float64Array(nRef);
    for (let i = 0; i < nRef; i++) refLrd[i] = lrd(ref, i * d, i);

    const out = new Float64Array(N);
    for (let row = 0; row < N; row++) {
      const { idx } = knn(flatInputs, row * numFeatures, -1);
      const lrdX = lrd(flatInputs, row * numFeatures, -1);
      if (idx.length === 0 || lrdX <= 0) { out[row] = 0; continue; }
      let sum = 0;
      for (const i of idx) sum += refLrd[i];
      out[row] = -((sum / idx.length) / lrdX);
    }
    return out;
  }

  return null;
}
