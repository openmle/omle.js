// Clustering model execution for the reference engine.

import type { Clustering } from '../ir.js';
import { tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';

export function executeClustering(
  clustering: Clustering,
  flatInputs: Float64Array,  // [N * numFeatures]
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): { labels: Int32Array; distances: Float64Array } {
  if (clustering.prototype) {
    return executePrototype(clustering.prototype, flatInputs, numFeatures, N, resolved);
  }
  if (clustering.gaussian_mixture) {
    return executeGMM(clustering.gaussian_mixture, flatInputs, numFeatures, N, resolved);
  }
  return { labels: new Int32Array(N), distances: new Float64Array(N) };
}

function executePrototype(
  proto: import('../ir.js').PrototypeClustering,
  flatInputs: Float64Array,
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): { labels: Int32Array; distances: Float64Array } {
  const { tensorIndex } = resolved;
  const centersTensor = resolveTensorValue(proto.centers, tensorIndex);
  if (!centersTensor) throw new Error('PrototypeClustering: centers tensor is missing');
  const centersData = tensorToData(centersTensor);
  const centers = centersData.data as Float64Array;
  const numClusters = centersTensor.type?.shape?.[0] ?? 1;
  const measure = proto.distance_measure ?? 'EUCLIDEAN';

  const labels = new Int32Array(N);
  const distances = new Float64Array(N);

  for (let row = 0; row < N; row++) {
    let bestCluster = 0;
    let bestDist = Infinity;

    for (let c = 0; c < numClusters; c++) {
      const dist = computeDistance(
        flatInputs, row * numFeatures, numFeatures,
        centers, c * numFeatures,
        measure,
      );
      if (dist < bestDist) {
        bestDist = dist;
        bestCluster = c;
      }
    }
    labels[row] = bestCluster;
    distances[row] = bestDist;
  }
  return { labels, distances };
}

function computeDistance(
  a: Float64Array, aOffset: number,
  n: number,
  b: Float64Array, bOffset: number,
  measure: string,
): number {
  switch (measure) {
    case 'EUCLIDEAN': {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = a[aOffset + i] - b[bOffset + i];
        sum += d * d;
      }
      return Math.sqrt(sum);
    }
    case 'SQUARED_EUCLIDEAN': {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = a[aOffset + i] - b[bOffset + i];
        sum += d * d;
      }
      return sum;
    }
    case 'MANHATTAN': {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += Math.abs(a[aOffset + i] - b[bOffset + i]);
      return sum;
    }
    case 'COSINE': {
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < n; i++) {
        dot += a[aOffset + i] * b[bOffset + i];
        normA += a[aOffset + i] ** 2;
        normB += b[bOffset + i] ** 2;
      }
      const denom = Math.sqrt(normA) * Math.sqrt(normB);
      return denom > 0 ? 1 - dot / denom : 1;
    }
    default: {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = a[aOffset + i] - b[bOffset + i];
        sum += d * d;
      }
      return Math.sqrt(sum);
    }
  }
}

function executeGMM(
  gmm: import('../ir.js').GaussianMixtureClustering,
  flatInputs: Float64Array,
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): { labels: Int32Array; distances: Float64Array } {
  const { tensorIndex } = resolved;
  const weightsTensor = resolveTensorValue(gmm.weights, tensorIndex);
  const meansTensor = resolveTensorValue(gmm.means, tensorIndex);
  const covsTensor = resolveTensorValue(gmm.covariances, tensorIndex);
  if (!weightsTensor) throw new Error('GaussianMixtureClustering: weights tensor is missing');
  if (!meansTensor) throw new Error('GaussianMixtureClustering: means tensor is missing');
  if (!covsTensor) throw new Error('GaussianMixtureClustering: covariances tensor is missing');
  const weights = tensorToData(weightsTensor).data as Float64Array;
  const means   = tensorToData(meansTensor).data as Float64Array;
  const covs    = tensorToData(covsTensor).data as Float64Array;
  const K = weights.length;
  const F = numFeatures;
  const covType = gmm.covariance_type ?? 'FULL';

  // Precompute per-component log-det contributions from the precision Cholesky diagonal.
  // For FULL: stored as precisions_cholesky_ [K, F, F], upper triangular.
  //   score_k = log(w_k) + sum(log(diag(L_k))) - 0.5 * ||(x-mu_k) @ L_k||^2
  // For DIAGONAL / SPHERICAL: stored as variance values [K, F] (tiled for spherical).
  //   score_k = log(w_k) - 0.5*(sum(log(var_kf)) + sum((x-mu_k)^2/var_kf))
  const logDet = new Float64Array(K);
  if (covType === 'FULL') {
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let f = 0; f < F; f++) s += Math.log(covs[k * F * F + f * F + f]);
      logDet[k] = s;
    }
  } else {
    // DIAGONAL or SPHERICAL: covs[k, f] is variance
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let f = 0; f < F; f++) s += Math.log(covs[k * F + f]);
      logDet[k] = -0.5 * s;
    }
  }

  const labels  = new Int32Array(N);
  const scores  = new Float64Array(N);

  for (let row = 0; row < N; row++) {
    let bestComp = 0, bestScore = -Infinity;
    for (let k = 0; k < K; k++) {
      let score = Math.log(weights[k] + 1e-300) + logDet[k];
      if (covType === 'FULL') {
        // y = (x - mu) @ L  (L is upper triangular precision Cholesky)
        let mah2 = 0;
        for (let j = 0; j < F; j++) {
          let yj = 0;
          for (let i = 0; i <= j; i++) {
            yj += (flatInputs[row * F + i] - means[k * F + i]) * covs[k * F * F + i * F + j];
          }
          mah2 += yj * yj;
        }
        score -= 0.5 * mah2;
      } else {
        // Diagonal/spherical: var = covs[k, f]
        let mah2 = 0;
        for (let f = 0; f < F; f++) {
          const d = flatInputs[row * F + f] - means[k * F + f];
          mah2 += d * d / covs[k * F + f];
        }
        score -= 0.5 * mah2;
      }
      if (score > bestScore) { bestScore = score; bestComp = k; }
    }
    labels[row] = bestComp;
    scores[row] = bestScore;
  }
  return { labels, distances: scores };
}
