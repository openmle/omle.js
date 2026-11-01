// Naive Bayes execution for the reference engine.

import type { NaiveBayes } from '../ir.js';
import { scalarToNumber } from '../ir.js';
import { applyPostTransform, tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';

export function executeNaiveBayes(
  nb: NaiveBayes,
  flatInputs: Float64Array,  // [N * numFeatures]
  numFeatures: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array {
  const { tensorIndex } = resolved;

  const priorsTensor = resolveTensorValue(nb.class_log_priors, tensorIndex);
  if (!priorsTensor) throw new Error('NaiveBayes: class_log_priors tensor is missing');
  const priorsData = tensorToData(priorsTensor);
  const logPriors = priorsData.data as Float64Array;
  const numClasses = logPriors.length;

  const logPosteriors = new Float64Array(N * numClasses);

  if (nb.gaussian) {
    const meansTensor = resolveTensorValue(nb.gaussian.means, tensorIndex);
    const varsTensor = resolveTensorValue(nb.gaussian.variances, tensorIndex);
    if (!meansTensor) throw new Error('GaussianNaiveBayes: means tensor is missing');
    if (!varsTensor) throw new Error('GaussianNaiveBayes: variances tensor is missing');
    const means = tensorToData(meansTensor).data as Float64Array;
    const vars = tensorToData(varsTensor).data as Float64Array;
    const eps = nb.gaussian.variance_epsilon != null ? scalarToNumber(nb.gaussian.variance_epsilon) : 0;

    for (let row = 0; row < N; row++) {
      for (let c = 0; c < numClasses; c++) {
        let logProb = logPriors[c];
        for (let f = 0; f < numFeatures; f++) {
          const x = flatInputs[row * numFeatures + f];
          const mu = means[c * numFeatures + f];
          const v = vars[c * numFeatures + f] + eps;
          logProb -= 0.5 * (Math.log(2 * Math.PI * v) + ((x - mu) ** 2) / v);
        }
        logPosteriors[row * numClasses + c] = logProb;
      }
    }
  } else if (nb.multinomial) {
    const flpTensor = resolveTensorValue(nb.multinomial.feature_log_prob, tensorIndex);
    if (!flpTensor) throw new Error('MultinomialNaiveBayes: feature_log_prob tensor is missing');
    const flp = tensorToData(flpTensor).data as Float64Array;

    for (let row = 0; row < N; row++) {
      for (let c = 0; c < numClasses; c++) {
        let logProb = logPriors[c];
        for (let f = 0; f < numFeatures; f++) {
          logProb += flatInputs[row * numFeatures + f] * flp[c * numFeatures + f];
        }
        logPosteriors[row * numClasses + c] = logProb;
      }
    }
  } else if (nb.bernoulli) {
    const flpTensor = resolveTensorValue(nb.bernoulli.feature_log_prob, tensorIndex);
    if (!flpTensor) throw new Error('BernoulliNaiveBayes: feature_log_prob tensor is missing');
    const flp = tensorToData(flpTensor).data as Float64Array;
    const binarizeThreshold = nb.bernoulli.binarize_threshold != null ? scalarToNumber(nb.bernoulli.binarize_threshold) : null;

    // Precompute log(1 - exp(feature_log_prob))
    const logNegProb = new Float64Array(flp.length);
    for (let i = 0; i < flp.length; i++) {
      logNegProb[i] = Math.log(1 - Math.exp(flp[i]) + 1e-10);
    }

    for (let row = 0; row < N; row++) {
      for (let c = 0; c < numClasses; c++) {
        let logProb = logPriors[c];
        for (let f = 0; f < numFeatures; f++) {
          const rawVal = flatInputs[row * numFeatures + f];
          const xi = binarizeThreshold !== null ? (rawVal > binarizeThreshold ? 1 : 0) : rawVal;
          logProb += xi * flp[c * numFeatures + f] + (1 - xi) * logNegProb[c * numFeatures + f];
        }
        logPosteriors[row * numClasses + c] = logProb;
      }
    }
  } else if (nb.categorical) {
    const clpTensor = resolveTensorValue(nb.categorical.category_log_prob, tensorIndex);
    if (!clpTensor) throw new Error('CategoricalNaiveBayes: category_log_prob tensor is missing');
    const clp = tensorToData(clpTensor).data as Float64Array;
    const catOffset = nb.categorical.category_offset ?? [];
    const catCount = nb.categorical.category_count ?? [];
    const rowSize = catOffset.length > 0
      ? (catOffset[catOffset.length - 1] + (catCount[catCount.length - 1] ?? 0))
      : numFeatures;

    for (let row = 0; row < N; row++) {
      for (let c = 0; c < numClasses; c++) {
        let logProb = logPriors[c];
        for (let f = 0; f < numFeatures; f++) {
          const catIdx = Math.round(flatInputs[row * numFeatures + f]);
          const offset = catOffset[f] ?? f;
          const count = catCount[f] ?? 1;
          const safeIdx = Math.max(0, Math.min(catIdx, count - 1));
          logProb += clp[c * rowSize + offset + safeIdx];
        }
        logPosteriors[row * numClasses + c] = logProb;
      }
    }
  }

  return applyPostTransform(logPosteriors, 'SOFTMAX', numClasses);
}
