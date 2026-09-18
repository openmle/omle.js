// SVM execution for the reference engine.

import type { SVM } from '../ir.js';
import { scalarToNumber } from '../ir.js';
import { applyPostTransform, matmulBias, tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';

export function executeSVM(
  svm: SVM,
  flatInputs: Float64Array,  // [N * numFeatures]
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array {
  const { tensorIndex } = resolved;

  if (svm.linear) {
    const coeffTensor = resolveTensorValue(svm.linear.coefficients, tensorIndex);
    if (!coeffTensor) throw new Error('LinearSVM: coefficients tensor is missing');
    const coeffData = tensorToData(coeffTensor);
    const coeffArr = coeffData.data as Float64Array;
    const coeffShape = coeffTensor.type?.shape ?? [];
    const outWidth = coeffShape.length === 2 ? coeffShape[0] : 1;

    let bias: Float64Array | null = null;
    if (svm.linear.intercept) {
      const interceptTensor = resolveTensorValue(svm.linear.intercept, tensorIndex);
      if (interceptTensor) bias = tensorToData(interceptTensor).data as Float64Array;
    }

    const scores = matmulBias(flatInputs, N, numFeatures, coeffArr, outWidth, bias);
    return applyPostTransform(scores, svm.post_transform, outWidth);
  }

  if (svm.kernel) {
    return executeKernelSVM(svm, flatInputs, numFeatures, N, resolved);
  }

  return new Float64Array(N);
}

function executeKernelSVM(
  svm: SVM,
  flatInputs: Float64Array,
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array {
  const k = svm.kernel!;
  const { tensorIndex } = resolved;

  const svTensor = resolveTensorValue(k.support_vectors, tensorIndex);
  const dualTensor = resolveTensorValue(k.dual_coefficients, tensorIndex);
  if (!svTensor) throw new Error('KernelSVM: support_vectors tensor is missing');
  if (!dualTensor) throw new Error('KernelSVM: dual_coefficients tensor is missing');

  const sv = tensorToData(svTensor).data as Float64Array;
  const dual = tensorToData(dualTensor).data as Float64Array;
  const svShape = svTensor.type?.shape ?? [];
  const totalSV = svShape[0] ?? 1;
  const dualShape = dualTensor.type?.shape ?? [];
  const dualRows = dualShape.length === 2 ? dualShape[0] : 1;
  const outWidth = dualRows > 1 ? dualRows : 1;

  let bias: Float64Array | null = null;
  if (k.intercept) {
    const interceptTensor = resolveTensorValue(k.intercept, tensorIndex);
    if (interceptTensor) bias = tensorToData(interceptTensor).data as Float64Array;
  }

  const gamma = k.gamma != null ? scalarToNumber(k.gamma) : (1 / numFeatures);
  const degree = k.degree ?? 3;
  const coef0 = k.coef0 != null ? scalarToNumber(k.coef0) : 0;
  const kernelType = k.kernel_type ?? 'RBF';

  const probATensor = k.prob_a ? resolveTensorValue(k.prob_a, tensorIndex) : null;
  const probBTensor = k.prob_b ? resolveTensorValue(k.prob_b, tensorIndex) : null;
  const nSupport = k.n_support ?? [];

  // Multiclass OVO Platt path (mirrors C++ predict_kernel_svm_multiclass).
  // Triggered when n_support covers 3+ classes and Platt params are present.
  if (nSupport.length > 2 && probATensor && probBTensor) {
    const probA = tensorToData(probATensor).data as Float64Array;
    const probB = tensorToData(probBTensor).data as Float64Array;
    const nClasses = nSupport.length;

    // sv_start[c] = index of first SV belonging to class c
    const svStart = new Int32Array(nClasses);
    for (let c = 1; c < nClasses; c++) svStart[c] = svStart[c - 1] + nSupport[c - 1];

    const result = new Float64Array(N * nClasses);
    for (let row = 0; row < N; row++) {
      const kernelVals = new Float64Array(totalSV);
      for (let s = 0; s < totalSV; s++) {
        kernelVals[s] = computeKernel(
          flatInputs, row * numFeatures, numFeatures,
          sv, s * numFeatures,
          kernelType, gamma, degree, coef0,
        );
      }

      // Compute K*(K-1)/2 pairwise decision values using libsvm OVO dual-coef layout.
      // For pair (ci, cj): dc_ci = dual[(cj-1)*nsv + svStart[ci]], dc_cj = dual[ci*nsv + svStart[cj]]
      const nPairs = (nClasses * (nClasses - 1)) / 2;
      const pairProbs = new Float64Array(nPairs);
      let pairIdx = 0;
      for (let ci = 0; ci < nClasses; ci++) {
        for (let cj = ci + 1; cj < nClasses; cj++) {
          let d = bias?.[pairIdx] ?? 0;
          const dciBase = (cj - 1) * totalSV + svStart[ci];
          for (let ki = 0; ki < nSupport[ci]; ki++) d += dual[dciBase + ki] * kernelVals[svStart[ci] + ki];
          const dcjBase = ci * totalSV + svStart[cj];
          for (let ki = 0; ki < nSupport[cj]; ki++) d += dual[dcjBase + ki] * kernelVals[svStart[cj] + ki];

          // Platt: rij = platt_sigmoid(d, A, B) = 1/(1+exp(A*d+B)) — NOT negated for multiclass
          const fApB = probA[pairIdx] * d + probB[pairIdx];
          let rij = fApB >= 0
            ? Math.exp(-fApB) / (1 + Math.exp(-fApB))
            : 1 / (1 + Math.exp(fApB));
          rij = Math.min(Math.max(rij, 1e-7), 1 - 1e-7);
          pairProbs[pairIdx] = rij;
          pairIdx++;
        }
      }

      const prob = wuCoupling(pairProbs, nClasses, 100);
      for (let c = 0; c < nClasses; c++) result[row * nClasses + c] = prob[c];
    }
    return result;
  }

  const scores = new Float64Array(N * outWidth);

  for (let row = 0; row < N; row++) {
    // Compute kernel values for all support vectors
    const kernelVals = new Float64Array(totalSV);
    for (let s = 0; s < totalSV; s++) {
      kernelVals[s] = computeKernel(
        flatInputs, row * numFeatures, numFeatures,
        sv, s * numFeatures,
        kernelType, gamma, degree, coef0,
      );
    }

    // decision = dual @ kernelVals + bias
    if (outWidth === 1) {
      let acc = bias?.[0] ?? 0;
      for (let s = 0; s < totalSV; s++) acc += dual[s] * kernelVals[s];
      scores[row] = acc;
    } else {
      for (let c = 0; c < outWidth; c++) {
        let acc = bias?.[c] ?? 0;
        for (let s = 0; s < totalSV; s++) acc += dual[c * totalSV + s] * kernelVals[s];
        scores[row * outWidth + c] = acc;
      }
    }
  }

  // Binary Platt scaling: p(class_1) = 1 - platt_sigmoid(-d, A, B), matching C++ convention.
  if (probATensor && probBTensor && outWidth === 1) {
    const probA = tensorToData(probATensor).data as Float64Array;
    const probB = tensorToData(probBTensor).data as Float64Array;
    const calibrated = new Float64Array(N * 2);
    for (let row = 0; row < N; row++) {
      const fApB = probA[0] * (-scores[row]) + probB[0];
      const plattSig = fApB >= 0
        ? Math.exp(-fApB) / (1 + Math.exp(-fApB))
        : 1 / (1 + Math.exp(fApB));
      const p = 1 - plattSig;
      calibrated[row * 2] = 1 - p;
      calibrated[row * 2 + 1] = p;
    }
    return calibrated;
  }

  return applyPostTransform(scores, svm.post_transform, outWidth);
}

// Multiclass probability estimation from pairwise calibrated probabilities.
// Implements libsvm's multiclass_probability (Wu, Lin & Weng, 2004).
// pairProbs[k] = P(class i | i vs j) for pairs enumerated i < j.
function wuCoupling(pairProbs: Float64Array, K: number, _maxIter: number): Float64Array {
  const maxIter = Math.max(100, K);
  const eps = 0.005 / K;
  const minProb = 1e-7;

  // Build the Q matrix (K x K)
  const Q = new Float64Array(K * K);
  let pairIdx = 0;
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      const rij = pairProbs[pairIdx];   // P(i | pair (i,j))
      const rji = 1 - rij;             // P(j | pair (i,j))
      Q[i * K + i] += rji * rji;
      Q[j * K + j] += rij * rij;
      Q[i * K + j] = -rij * rji;
      Q[j * K + i] = -rij * rji;
      pairIdx++;
    }
  }

  // Initialize uniformly
  const p = new Float64Array(K).fill(1 / K);
  const Qp = new Float64Array(K);

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute Qp and pQp
    let pQp = 0;
    for (let t = 0; t < K; t++) {
      Qp[t] = 0;
      for (let j = 0; j < K; j++) Qp[t] += Q[t * K + j] * p[j];
      pQp += p[t] * Qp[t];
    }
    // Convergence check
    let maxErr = 0;
    for (let t = 0; t < K; t++) {
      const e = Math.abs(Qp[t] - pQp);
      if (e > maxErr) maxErr = e;
    }
    if (maxErr < eps) break;
    // Coordinate descent update (libsvm convention)
    for (let t = 0; t < K; t++) {
      const diff = (-Qp[t] + pQp) / Q[t * K + t];
      p[t] += diff;
      pQp = (pQp + diff * (diff * Q[t * K + t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
      for (let j = 0; j < K; j++) {
        Qp[j] = (Qp[j] + diff * Q[t * K + j]) / (1 + diff);
        p[j] /= (1 + diff);
      }
      if (p[t] < minProb) p[t] = minProb;
    }
  }
  return p;
}

function computeKernel(
  x: Float64Array, xOffset: number,
  n: number,
  sv: Float64Array, svOffset: number,
  kernelType: string,
  gamma: number, degree: number, coef0: number,
): number {
  switch (kernelType) {
    case 'LINEAR': {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += x[xOffset + i] * sv[svOffset + i];
      return dot;
    }
    case 'RBF': {
      let sqDist = 0;
      for (let i = 0; i < n; i++) {
        const d = x[xOffset + i] - sv[svOffset + i];
        sqDist += d * d;
      }
      return Math.exp(-gamma * sqDist);
    }
    case 'POLY': {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += x[xOffset + i] * sv[svOffset + i];
      return Math.pow(gamma * dot + coef0, degree);
    }
    case 'SIGMOID': {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += x[xOffset + i] * sv[svOffset + i];
      return Math.tanh(gamma * dot + coef0);
    }
    default: {
      let sqDist = 0;
      for (let i = 0; i < n; i++) {
        const d = x[xOffset + i] - sv[svOffset + i];
        sqDist += d * d;
      }
      return Math.exp(-gamma * sqDist);
    }
  }
}
