// Neural network (MLP) execution for the reference engine.

import type { NeuralNetwork } from '../ir.js';
import { applyActivation, matmulBias, tensorToData } from './ops.js';
import type { ResolvedModel } from '../resolve.js';
import { resolveTensorValue } from '../resolve.js';


// Evaluates a classical feed-forward MLP: y = activation(W @ x + b) per layer.
export function executeNeuralNetwork(
  nn: NeuralNetwork,
  flatInputs: Float64Array,  // [N * inputWidth]
  inputWidth: number,
  N: number,
  resolved: ResolvedModel,
): Float64Array {
  const { tensorIndex } = resolved;
  const layers = nn.layers ?? [];

  if (layers.length === 0) return flatInputs;

  let current = flatInputs;
  let currentWidth = inputWidth;

  for (const layer of layers) {
    const wTensor = resolveTensorValue(layer.weights, tensorIndex);
    if (!wTensor) throw new Error(`DenseLayer "${layer.name ?? '?'}": weights tensor is missing`);
    const wData = tensorToData(wTensor);
    const weights = wData.data as Float64Array;
    const wShape = wTensor.type?.shape ?? [];
    const outWidth = wShape[0] ?? 1;
    const inWidth = wShape[1] ?? currentWidth;

    let bias: Float64Array | null = null;
    if (layer.bias) {
      const biasTensor = resolveTensorValue(layer.bias, tensorIndex);
      if (biasTensor) bias = tensorToData(biasTensor).data as Float64Array;
    }

    const preActivation = matmulBias(current, N, inWidth, weights, outWidth, bias);
    current = applyActivation(preActivation, layer.activation, outWidth);
    currentWidth = outWidth;
  }

  return current;
}
