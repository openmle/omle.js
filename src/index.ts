// omle.js — Public API
//
// Usage:
//   import { fromJSON, validate, Engine } from '@openmle/omle.js';
//   const model = fromJSON(jsonString);
//   const result = validate(model);
//   const engine = new Engine(model);
//   const outputs = engine.run({ X: { dtype: 'FLOAT64', shape: [1, 4], data: new Float64Array([...]) } });

export * from './ir.js';
export * from './io.js';
export { fromProtoBinary } from './proto.js';
export * from './resolve.js';
export * from './validate.js';

export { Engine } from './engine/executor.js';
export { validateInputs } from './engine/validate_inputs.js';
export type { InputIssue, InputValidationResult } from './engine/validate_inputs.js';
export type { InferenceInput, InferenceOutput, SerializedTensor, StepSnapshot, SteppedResult, VerifyResult, CaseVerifyResult, OutputVerifyResult } from './engine/executor.js';
export type {
  ModelExplain, TreeEnsembleExplain, TreeExplainItem, TreePathStep,
  LinearExplain, LinearContribution,
  NaiveBayesExplain, NaiveBayesFeatureContrib,
  ClusteringExplain, CentroidDistance,
} from './engine/explain.js';
export type { TensorData } from './engine/ops.js';
export { tensorToData, applyPostTransform } from './engine/ops.js';
