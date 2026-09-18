// Linear model execution for the reference engine.

import type { Linear } from '../ir.js';
import { applyPostTransform, matmulBias, tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';

// Linear: score = X @ coeff^T + intercept, then post_transform
export function executeLinear(
  linear: Linear,
  flatInputs: Float64Array,  // [N * numSlots]
  numSlots: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array {
  const { tensorIndex } = resolved;

  const coeffTensor = resolveTensorValue(linear.coefficients, tensorIndex);
  if (!coeffTensor) throw new Error('Linear: coefficients tensor is missing');
  const coeffData = tensorToData(coeffTensor);
  const coeffArr = coeffData.data as Float64Array;
  const coeffShape = coeffTensor.type?.shape ?? [];

  // Determine output width
  let outWidth: number;
  let weights: Float64Array;

  if (coeffShape.length === 2) {
    // [outWidth, numSlots] — multiclass or multi-output
    outWidth = coeffShape[0];
    weights = coeffArr;
  } else {
    // [numSlots] — single output
    outWidth = 1;
    weights = coeffArr;
  }

  // Build bias
  let bias: Float64Array | null = null;
  if (linear.intercept) {
    const interceptTensor = resolveTensorValue(linear.intercept, tensorIndex);
    if (interceptTensor) bias = tensorToData(interceptTensor).data as Float64Array;
  }

  const scores = matmulBias(flatInputs, N, numSlots, weights, outWidth, bias);
  return applyPostTransform(scores, linear.post_transform, outWidth);
}
