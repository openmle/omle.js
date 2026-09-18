// Reference execution engine for OMLE models.
//
// Executes nodes in topological order, building a namespace of named TensorData values.
// Supports: Tree, TreeEnsemble, Linear, NeuralNetwork, NaiveBayes, Clustering, SVM,
//           and CompositeNode (via recursive execution).

import type { OMLEModel, Node, Tensor, VerificationCase, TensorRef } from '../ir.js';
import { scalarToNumber } from '../ir.js';
import type { ResolvedModel } from '../resolve.js';
import { resolve, expandNodeInputs, expandFeatures, resolveTensorValue } from '../resolve.js';
import type { TensorData } from './ops.js';
import { tensorToData, tensorCols } from './ops.js';
import { executeTree, executeTreeEnsemble } from './tree.js';
import { executeLinear } from './linear.js';
import { executeNeuralNetwork } from './nn.js';
import { executeNaiveBayes } from './naive_bayes.js';
import { executeClustering } from './clustering.js';
import { executeSVM } from './svm.js';
import { computeExplain } from './explain.js';
import type { ModelExplain } from './explain.js';
import {
  executeTakeSlots, executeConcat, executeDerive,
  executeBinarizer, executeMinMaxScaler, executeMaxAbsScaler,
  executeRobustScaler, executeNormalizer,
  executeOneHotEncoder, executeBucketizer,
  executePolynomialFeatures, executePowerTransformer,
  executeQuantileTransformer, executeSplineTransformer,
  executeTruncatedSVD, executePCA, executeImputer, executeWeightedSum, executeNormContinuous,
  executeTokenizer, executeRegexTokenizer,
  executeNGram, executeStopWordsRemover, executeCountVectorizer,
  executeHashingVectorizer, executeTfIdfTransformer, executeWord2Vec,
} from './preprocess.js';

// ── Public API ────────────────────────────────────────────────────────────────

// Accepts TensorData, IR Tensor, plain number/array, or 2-D number[][] per input name.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type InferenceInput = Record<string, any>;

export interface InferenceOutput {
  [name: string]: TensorData;
}

export interface SerializedTensor {
  dtype: string;
  shape: number[];
  /** Capped at 50 elements for display; null if value was not produced */
  data: (number | string | boolean)[] | null;
}

export interface StepSnapshot {
  nodeName: string;
  nodeId: string;    // "node:<name>"
  inputs: Record<string, SerializedTensor>;
  outputs: Record<string, SerializedTensor>;
  /** Warnings or caught errors during this step */
  warnings: string[];
  /** Wall-clock execution time for this node in milliseconds */
  durationMs: number;
  /** Structured model body breakdown — present for tree ensemble, linear, naive bayes, clustering */
  explain?: ModelExplain;
}

export interface SteppedResult {
  output: InferenceOutput;
  steps: StepSnapshot[];
}

export interface OutputVerifyResult {
  name: string;
  pass: boolean;
  maxAbsErr: number;
  maxRelErr: number;
  atol: number;
  rtol: number;
}

export interface CaseVerifyResult {
  index: number;
  pass: boolean;
  outputs: OutputVerifyResult[];
}

export interface VerifyResult {
  pass: boolean;
  cases: CaseVerifyResult[];
}

export class Engine {
  private resolved: ResolvedModel;

  constructor(model: OMLEModel) {
    this.resolved = resolve(model);
  }

  run(inputs: InferenceInput): InferenceOutput {
    const { resolved } = this;
    const model = resolved.model;

    // Normalize all inputs to TensorData so downstream code always sees .shape/.data
    const namespace = new Map<string, TensorData>();
    for (const [name, val] of Object.entries(inputs)) {
      namespace.set(name, normalizeInput(val));
    }

    const N = inferBatchSize(namespace);
    expandSchemaFeatures(model, namespace, N);

    // Use topologically sorted order computed during resolve()
    executeNodes(resolved.executionOrder, namespace, N, resolved);

    const outputs: InferenceOutput = {};
    for (const outSpec of model.outputs ?? []) {
      const td = namespace.get(outSpec.name);
      if (td) outputs[outSpec.name] = td;
    }
    return outputs;
  }

  verify(): VerifyResult {
    const { resolved } = this;
    const model = resolved.model;
    const ver = model.verification;
    if (!ver?.cases?.length) return { pass: true, cases: [] };

    const atol = ver.tolerance?.atol != null ? scalarToNumber(ver.tolerance.atol) : 1e-6;
    const rtol = ver.tolerance?.rtol != null ? scalarToNumber(ver.tolerance.rtol) : 1e-5;

    const tensorIndex = resolved.tensorIndex;
    const modelInputs = model.inputs ?? [];

    const caseResults: CaseVerifyResult[] = [];

    for (let ci = 0; ci < ver.cases.length; ci++) {
      const vc: VerificationCase = ver.cases[ci]!;
      const inferInputs: InferenceInput = {};
      const vcInputs: TensorRef[] = vc.inputs ?? [];
      for (let i = 0; i < vcInputs.length; i++) {
        const ref: TensorRef = vcInputs[i]!;
        const entry = tensorIndex.get(ref.id);
        if (!entry?.dense) continue;
        const name = entry.dense.name ?? modelInputs[i]?.name ?? ref.id;
        inferInputs[name] = tensorToData(entry.dense);
      }

      const output = this.run(inferInputs);
      const outputResults: OutputVerifyResult[] = [];

      const vcExpected: TensorRef[] = vc.expected_outputs ?? [];
      for (let i = 0; i < vcExpected.length; i++) {
        const ref: TensorRef = vcExpected[i]!;
        const entry = tensorIndex.get(ref.id);
        if (!entry?.dense) continue;
        const outName = entry.dense.name ?? model.outputs?.[i]?.name ?? ref.id;
        const actual = output[outName];
        if (!actual) {
          outputResults.push({ name: outName, pass: false, maxAbsErr: Infinity, maxRelErr: Infinity, atol, rtol });
          continue;
        }
        const expected = tensorToData(entry.dense);
        const { pass, maxAbsErr, maxRelErr } = compareTensors(actual, expected, atol, rtol);
        outputResults.push({ name: outName, pass, maxAbsErr, maxRelErr, atol, rtol });
      }

      const casePass = outputResults.length > 0 && outputResults.every(r => r.pass);
      caseResults.push({ index: ci, pass: casePass, outputs: outputResults });
    }

    return { pass: caseResults.every(r => r.pass), cases: caseResults };
  }

  runWithSteps(inputs: InferenceInput): SteppedResult {
    const { resolved } = this;
    const model = resolved.model;

    const namespace = new Map<string, TensorData>();
    for (const [name, val] of Object.entries(inputs)) {
      namespace.set(name, normalizeInput(val));
    }
    const N = inferBatchSize(namespace);
    expandSchemaFeatures(model, namespace, N);
    const steps: StepSnapshot[] = [];

    for (const node of resolved.executionOrder) {
      const inputNames = [...new Set(expandNodeInputs(node.inputs ?? []))];
      const inputsBefore: Record<string, SerializedTensor> = {};
      for (const name of inputNames) {
        inputsBefore[name] = serializeTensor(namespace.get(name));
      }

      const warnings: string[] = [];
      const t0 = performance.now();
      try {
        executeNode(node, namespace, N, resolved);
      } catch (e) {
        warnings.push(String(e));
      }
      const durationMs = performance.now() - t0;

      const outputsAfter: Record<string, SerializedTensor> = {};
      for (const out of node.outputs ?? []) {
        const td = namespace.get(out.name);
        outputsAfter[out.name] = serializeTensor(td);
        if (!td && warnings.length === 0) warnings.push(`Output "${out.name}" was not produced`);
      }

      const explain = computeExplain(node, namespace, resolved);
      steps.push({ nodeName: node.name, nodeId: `node:${node.name}`, inputs: inputsBefore, outputs: outputsAfter, warnings, durationMs, explain });
    }

    const output: InferenceOutput = {};
    for (const outSpec of model.outputs ?? []) {
      const td = namespace.get(outSpec.name);
      if (td) output[outSpec.name] = td;
    }
    return { output, steps };
  }

  get model(): OMLEModel {
    return this.resolved.model;
  }
}

// ── Schema feature expansion ──────────────────────────────────────────────────

// Expands model_schema features with a source+index (e.g. columns of a matrix input X)
// into individual named tensors in the namespace, so nodes can address them by name.
function expandSchemaFeatures(
  model: OMLEModel,
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const schema = model.model_schema;
  if (!schema?.features) return;
  for (const f of expandFeatures(schema.features)) {
    if (!f.name || f.source === undefined) continue;
    if (namespace.has(f.name)) continue;
    const src = namespace.get(f.source);
    if (!src) continue;
    const cols = tensorCols(src);
    const col = f.index ?? 0;  // proto omits default 0, so undefined means column 0
    if (col < 0 || col >= cols) continue;
    const colData = new Float64Array(N);
    const data = src.data as Float64Array;
    for (let row = 0; row < N; row++) colData[row] = data[row * cols + col];
    namespace.set(f.name, { dtype: 'FLOAT64', shape: [N], data: colData });
  }
}

// ── Tensor comparison ─────────────────────────────────────────────────────────

function compareTensors(
  actual: TensorData,
  expected: TensorData,
  atol: number,
  rtol: number,
): { pass: boolean; maxAbsErr: number; maxRelErr: number } {
  const a = actual.data;
  const e = expected.data;
  if (a.length !== e.length) return { pass: false, maxAbsErr: Infinity, maxRelErr: Infinity };
  let maxAbsErr = 0, maxRelErr = 0, pass = true;
  for (let i = 0; i < a.length; i++) {
    const ai = Number(a[i]), ei = Number(e[i]);
    const absErr = Math.abs(ai - ei);
    const relErr = Math.abs(ei) > 0 ? absErr / Math.abs(ei) : absErr;
    if (absErr > maxAbsErr) maxAbsErr = absErr;
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (absErr > atol + rtol * Math.abs(ei)) pass = false;
  }
  return { pass, maxAbsErr, maxRelErr };
}

// ── Input normalization ───────────────────────────────────────────────────────

function normalizeInput(val: unknown): TensorData {
  if (val === null || val === undefined) {
    return { dtype: 'FLOAT64', shape: [1], data: new Float64Array([0]) };
  }

  // Already TensorData — has .data / .shape / .dtype in internal format
  if (typeof val === 'object' && !Array.isArray(val)) {
    const obj = val as Record<string, unknown>;

    if ('data' in obj && 'shape' in obj && 'dtype' in obj) {
      const td = obj as unknown as TensorData;
      // data may be a plain number[] if the caller built this from JSON — but preserve string/bool arrays
      if (Array.isArray(td.data)) {
        if (td.dtype === 'STRING' || td.dtype === 'BOOL') return td;
        return { dtype: td.dtype, shape: td.shape, data: new Float64Array(td.data as unknown as number[]) };
      }
      return td;
    }

    // IR Tensor format (float64_data / float32_data / int32_data / …)
    if (obj['float64_data'] || obj['float32_data'] || obj['int32_data'] ||
        obj['int64_data'] || obj['string_data'] || obj['bool_data'] || obj['raw_data']) {
      return tensorToData(obj as Tensor);
    }
  }

  // Scalar
  if (typeof val === 'number') {
    return { dtype: 'FLOAT64', shape: [1], data: new Float64Array([val]) };
  }
  if (typeof val === 'boolean') {
    return { dtype: 'BOOL', shape: [1], data: [val] };
  }
  if (typeof val === 'string') {
    return { dtype: 'STRING', shape: [1], data: [val] };
  }

  if (Array.isArray(val)) {
    if (val.length === 0) return { dtype: 'FLOAT64', shape: [0], data: new Float64Array(0) };

    // 2-D array
    if (Array.isArray(val[0])) {
      const rows = val.length;
      const cols = (val[0] as unknown[]).length;
      const flat = new Float64Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const row = val[r] as number[];
        for (let c = 0; c < cols; c++) flat[r * cols + c] = row[c];
      }
      return { dtype: 'FLOAT64', shape: [rows, cols], data: flat };
    }

    // 1-D array
    const first = val[0];
    if (typeof first === 'string') return { dtype: 'STRING', shape: [val.length], data: val as string[] };
    if (typeof first === 'boolean') return { dtype: 'BOOL', shape: [val.length], data: val as boolean[] };
    return { dtype: 'FLOAT64', shape: [val.length], data: new Float64Array(val as number[]) };
  }

  return { dtype: 'FLOAT64', shape: [1], data: new Float64Array([0]) };
}

// ── Node execution ────────────────────────────────────────────────────────────

function executeNodes(
  nodes: Node[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
) {
  for (const node of nodes) {
    executeNode(node, namespace, N, resolved);
  }
}

function executeNode(
  node: Node,
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
) {
  if (node.composite) {
    executeComposite(node, namespace, N, resolved);
    return;
  }

  // Build flat slot space from node inputs
  const inputNames = expandNodeInputs(node.inputs ?? []);
  const { flatInputs, numSlots } = buildFlatSlotSpace(inputNames, namespace, N);

  let outputData: Float64Array | null = null;
  let outputWidth = 1;

  if (node.tree) {
    outputData = executeTree(node.tree, flatInputs, numSlots, N, resolved);
    outputWidth = outputData.length / N;
  } else if (node.tree_ensemble) {
    outputData = executeTreeEnsemble(node.tree_ensemble, flatInputs, numSlots, N, resolved);
    outputWidth = outputData.length / N;
  } else if (node.linear) {
    const coeffTensor = resolveTensorValue(node.linear.coefficients, resolved.tensorIndex);
    const coeffShape = coeffTensor?.type?.shape ?? [];
    outputWidth = coeffShape.length === 2 ? (coeffShape[0] ?? 1) : 1;
    outputData = executeLinear(node.linear, flatInputs, numSlots, N, resolved);
  } else if (node.neural_network) {
    const layers = node.neural_network.layers ?? [];
    const lastLayer = layers.length > 0 ? layers[layers.length - 1] : undefined;
    if (lastLayer) {
      const wTensor = resolveTensorValue(lastLayer.weights, resolved.tensorIndex);
      const wShape = wTensor?.type?.shape ?? [];
      outputWidth = wShape[0] ?? 1;
    }
    outputData = executeNeuralNetwork(node.neural_network, flatInputs, numSlots, N, resolved);
  } else if (node.naive_bayes) {
    const priorTensor = resolveTensorValue(node.naive_bayes.class_log_priors, resolved.tensorIndex);
    const priorShape = priorTensor?.type?.shape ?? [];
    outputWidth = priorShape[0] ?? 1;
    outputData = executeNaiveBayes(node.naive_bayes, flatInputs, numSlots, N, resolved);
  } else if (node.svm) {
    outputData = executeSVM(node.svm, flatInputs, numSlots, N, resolved);
    outputWidth = outputData.length / N;
  } else if (node.clustering) {
    const { labels, distances } = executeClustering(
      node.clustering, flatInputs, numSlots, N, resolved);
    // Publish cluster label and distance as separate outputs
    publishClusteringOutputs(node, labels, distances, namespace, N);
    return;
  } else if (node.op === 'OrdinalEncoder' || node.op === 'ordinal_encoder' ||
             node.op === 'TargetEncoder' || node.op === 'CountEncoder' ||
             node.op === 'WOEEncoder' || node.op === 'JamesSteinEncoder' ||
             node.op === 'MEstimateEncoder' || node.op === 'QuantileEncoder' ||
             node.op === 'CatBoostEncoder' || node.op === 'LeaveOneOutEncoder') {
    executeOrdinalEncoder(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'LabelEncoder') {
    executeLabelEncoder(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'StandardScaler') {
    executeStandardScaler(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'SAMMEVote') {
    executeSAMMEVote(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'WeightedMedian') {
    executeWeightedMedian(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'Average') {
    executeAverage(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'ArgMax') {
    executeArgMax(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'SoftVote') {
    executeSoftVote(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'MajorityVote') {
    executeMajorityVote(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'TakeSlots') {
    executeTakeSlots(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'Concat') {
    executeConcat(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'Derive') {
    executeDerive(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'Binarizer') {
    executeBinarizer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'MinMaxScaler') {
    executeMinMaxScaler(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'MaxAbsScaler') {
    executeMaxAbsScaler(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'RobustScaler') {
    executeRobustScaler(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'Normalizer') {
    executeNormalizer(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'OneHotEncoder') {
    executeOneHotEncoder(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'Bucketizer') {
    executeBucketizer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'PolynomialFeatures') {
    executePolynomialFeatures(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'PowerTransformer') {
    executePowerTransformer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'QuantileTransformer') {
    executeQuantileTransformer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'SplineTransformer') {
    executeSplineTransformer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'TruncatedSVD') {
    executeTruncatedSVD(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'PCA') {
    executePCA(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'Imputer') {
    executeImputer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'WeightedSum') {
    executeWeightedSum(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'NormContinuous') {
    executeNormContinuous(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'Tokenizer') {
    executeTokenizer(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'RegexTokenizer') {
    executeRegexTokenizer(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'NGram') {
    executeNGram(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'StopWordsRemover') {
    executeStopWordsRemover(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'CountVectorizer') {
    executeCountVectorizer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'HashingVectorizer') {
    executeHashingVectorizer(node, inputNames, namespace, N);
    return;
  } else if (node.op === 'TfIdfTransformer') {
    executeTfIdfTransformer(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'Word2Vec') {
    executeWord2Vec(node, inputNames, namespace, N, resolved);
    return;
  } else if (node.op === 'KNN') {
    executeKNN(node, inputNames, namespace, N, resolved);
    return;
  } else {
    // Generic operator — pass through first input as-is (placeholder)
    if (inputNames.length > 0) {
      const td = namespace.get(inputNames[0]);
      for (const out of node.outputs ?? []) {
        if (td) namespace.set(out.name, td);
      }
    }
    return;
  }

  if (outputData) {
    publishNumericOutputs(node, outputData, outputWidth, N, namespace);
  }
}

// ── Composite node execution ──────────────────────────────────────────────────

function executeComposite(
  node: Node,
  parentNamespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
) {
  const composite = node.composite!;
  const localNS = new Map<string, TensorData>();

  // Inherit inputs from parent namespace
  const inputNames = expandNodeInputs(node.inputs ?? []);
  for (const name of inputNames) {
    const td = parentNamespace.get(name);
    if (td) localNS.set(name, td);
  }

  // Apply input aliases
  for (const alias of composite.input_aliases ?? []) {
    const td = localNS.get(alias.from_name);
    if (td) {
      localNS.delete(alias.from_name);
      localNS.set(alias.to_name, td);
    }
  }

  // Expand schema features into the local namespace so that name-keyed TakeSlots
  // nodes (e.g. from ColumnTransformer) can resolve individual column names.
  expandSchemaFeatures(resolved.model, localNS, N);

  // Execute internal nodes
  executeNodes(composite.nodes ?? [], localNS, N, resolved);

  // Publish outputs back to parent
  for (const out of node.outputs ?? []) {
    let sourceName = out.name;
    for (const alias of composite.output_aliases ?? []) {
      if (alias.to_name === out.name) { sourceName = alias.from_name; break; }
    }
    const td = localNS.get(sourceName);
    if (td) parentNamespace.set(out.name, td);
  }
}

// ── OrdinalEncoder ────────────────────────────────────────────────────────────

function executeOrdinalEncoder(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const attrs = node.attributes ?? [];
  const catAttr = attrs.find(a => a.name === 'categories');
  const offAttr = attrs.find(a => a.name === 'category_offsets');
  if (!catAttr?.tensor) return;

  const categories = tensorToData(catAttr.tensor).data as string[];
  // When offsets are absent (e.g. Spark StringIndexer single-feature), default to one feature spanning all categories.
  const offsets: number[] = offAttr?.tensor
    ? Array.from(tensorToData(offAttr.tensor).data as Float64Array)
    : [0, categories.length];

  // Optional: target-encoded float values per category (e.g. CatBoost encoder)
  const encAttr = attrs.find(a => a.name === 'encoded_values');
  const defAttr = attrs.find(a => a.name === 'default_values');
  const encodedValues = encAttr?.tensor ? tensorToData(encAttr.tensor).data as Float64Array : null;
  const defaultValues = defAttr?.tensor ? tensorToData(defAttr.tensor).data as Float64Array : null;

  // numFeatures comes from offsets when a single matrix input is provided,
  // or from the number of named inputs when each feature is a separate column.
  const numFeatures = inputNames.length > 1
    ? inputNames.length
    : (offsets.length > 1 ? offsets.length - 1 : 1);
  const out = new Float64Array(N * numFeatures);

  // Single matrix input: all features share one TensorData with column stride numFeatures.
  const singleTd = inputNames.length === 1 ? namespace.get(inputNames[0]) : undefined;
  const singleData = singleTd?.data ?? null;
  const isNumericInput = singleData instanceof Float32Array || singleData instanceof Float64Array;

  for (let fi = 0; fi < numFeatures; fi++) {
    const start = offsets[fi] ?? 0;
    const end = offsets[fi + 1] ?? categories.length;

    if (encodedValues) {
      const catStrMap = new Map<string, number>();
      const catNumMap = new Map<number, number>();
      for (let ci = start; ci < end; ci++) {
        catStrMap.set(categories[ci], encodedValues[ci]);
        const n = Number(categories[ci]);
        if (!isNaN(n)) catNumMap.set(n, encodedValues[ci]);
      }
      const defaultVal = defaultValues?.[fi] ?? NaN;
      for (let row = 0; row < N; row++) {
        const raw = singleData ? singleData[row * numFeatures + fi]
                               : (namespace.get(inputNames[fi])?.data as unknown[])?.[row];
        const v = isNumericInput
          ? (catNumMap.get(raw as number) ?? defaultVal)
          : (catStrMap.get(String(raw)) ?? defaultVal);
        out[row * numFeatures + fi] = v;
      }
    } else {
      const catStrMap = new Map<string, number>();
      const catNumMap = new Map<number, number>();
      for (let ci = start; ci < end; ci++) {
        catStrMap.set(categories[ci], ci - start);
        const n = Number(categories[ci]);
        if (!isNaN(n)) catNumMap.set(n, ci - start);
      }
      for (let row = 0; row < N; row++) {
        const raw = singleData ? singleData[row * numFeatures + fi]
                               : (namespace.get(inputNames[fi])?.data as unknown[])?.[row];
        const v = isNumericInput
          ? (catNumMap.get(raw as number) ?? NaN)
          : (catStrMap.get(String(raw)) ?? NaN);
        out[row * numFeatures + fi] = v;
      }
    }
  }

  const outputs = node.outputs ?? [];
  if (outputs.length > 0) {
    namespace.set(outputs[0].name, { dtype: 'FLOAT64', shape: [N, numFeatures], data: out });
  }
}

// ── LabelEncoder ─────────────────────────────────────────────────────────────

function executeLabelEncoder(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const attrs = node.attributes ?? [];
  const labelsAttr = attrs.find(a => a.name === 'labels');
  const offsetsAttr = attrs.find(a => a.name === 'label_offsets');
  if (!labelsAttr?.tensor || !offsetsAttr?.tensor) return;

  const labels = tensorToData(labelsAttr.tensor).data as string[];
  const offsets = Array.from(tensorToData(offsetsAttr.tensor).data as Float64Array);

  const numFeatures = inputNames.length;
  const outputs = node.outputs ?? [];
  const onePerOutput = outputs.length >= numFeatures;

  const combined = onePerOutput ? null : new Float64Array(N * numFeatures);

  for (let fi = 0; fi < numFeatures; fi++) {
    const td = namespace.get(inputNames[fi]);
    const start = offsets[fi] ?? 0;
    const end = offsets[fi + 1] ?? labels.length;
    const labelMap = new Map<string, number>();
    for (let ci = start; ci < end; ci++) labelMap.set(labels[ci], ci - start);

    const strData = (td?.data ?? []) as string[];
    if (onePerOutput) {
      const col = new Float64Array(N);
      for (let row = 0; row < N; row++) col[row] = labelMap.get(strData[row]) ?? NaN;
      namespace.set(outputs[fi].name, { dtype: 'FLOAT64', shape: [N], data: col });
    } else {
      for (let row = 0; row < N; row++) {
        combined![row * numFeatures + fi] = labelMap.get(strData[row]) ?? NaN;
      }
    }
  }

  if (!onePerOutput && outputs.length > 0) {
    namespace.set(outputs[0].name, { dtype: 'FLOAT64', shape: [N, numFeatures], data: combined! });
  }
}

// ── StandardScaler ────────────────────────────────────────────────────────────

function executeStandardScaler(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const attrs = node.attributes ?? [];
  const meanAttr = attrs.find(a => a.name === 'mean');
  const scaleAttr = attrs.find(a => a.name === 'scale');
  if (!meanAttr?.tensor && !scaleAttr?.tensor) return;

  const meanRaw = meanAttr?.tensor ? (tensorToData(meanAttr.tensor).data as Float64Array) : null;
  const scaleRaw = scaleAttr?.tensor ? (tensorToData(scaleAttr.tensor).data as Float64Array) : null;

  // Collect all input tensors as a combined matrix [N, totalCols]
  let totalCols = 0;
  for (const name of inputNames) {
    const td = namespace.get(name);
    totalCols += td ? tensorCols(td) : 0;
  }

  const out = new Float64Array(N * totalCols);
  let colOffset = 0;
  for (const name of inputNames) {
    const td = namespace.get(name);
    if (!td) continue;
    const cols = tensorCols(td);
    const data = td.data as Float64Array;
    for (let row = 0; row < N; row++) {
      for (let c = 0; c < cols; c++) {
        const globalCol = colOffset + c;
        const mu = meanRaw?.[globalCol] ?? 0;
        const sigma = scaleRaw?.[globalCol] ?? 1;
        out[row * totalCols + globalCol] = (data[row * cols + c] - mu) / sigma;
      }
    }
    colOffset += cols;
  }

  const outputs = node.outputs ?? [];
  if (outputs.length > 0) {
    namespace.set(outputs[0].name, { dtype: 'FLOAT64', shape: [N, totalCols], data: out });
  }
}

// ── SAMMEVote ─────────────────────────────────────────────────────────────────

function executeSAMMEVote(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const attrs = node.attributes ?? [];
  const weightsAttr = attrs.find(a => a.name === 'weights');
  const nClassesAttr = attrs.find(a => a.name === 'n_classes');

  const weights: number[] = weightsAttr?.float64s ?? [];
  const K = nClassesAttr?.i ?? 2;

  // SAMME discrete: each estimator contributes +alpha to predicted class, -alpha/(K-1) to others.
  // Then scores /= total_weight, prob = exp(score / (K-1)), normalize rows.
  const totalWeight = weights.reduce((s, w) => s + w, 0) || 1;
  const scores = new Float64Array(N * K);
  for (let t = 0; t < inputNames.length; t++) {
    const alpha = weights[t] ?? 1;
    const td = namespace.get(inputNames[t]);
    if (!td) continue;
    const preds = td.data as Float64Array;
    for (let row = 0; row < N; row++) {
      const cls = Math.round(preds[row]);
      for (let k = 0; k < K; k++) {
        scores[row * K + k] += k === cls ? alpha : -alpha / (K - 1);
      }
    }
  }

  for (let i = 0; i < scores.length; i++) scores[i] /= totalWeight;

  const factor = 1.0 / (K - 1);
  const prob = new Float64Array(N * K);
  for (let row = 0; row < N; row++) {
    let sum = 0;
    for (let k = 0; k < K; k++) {
      prob[row * K + k] = Math.exp(factor * scores[row * K + k]);
      sum += prob[row * K + k];
    }
    for (let k = 0; k < K; k++) prob[row * K + k] /= sum;
  }
  const nClasses = K;

  // Argmax for prediction
  const pred = new Float64Array(N);
  for (let row = 0; row < N; row++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let k = 0; k < nClasses; k++) {
      if (prob[row * nClasses + k] > maxVal) { maxVal = prob[row * nClasses + k]; maxIdx = k; }
    }
    pred[row] = maxIdx;
  }

  const outputs = node.outputs ?? [];
  for (const out of outputs) {
    if (out.role === 'PROBABILITY') {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N, nClasses], data: prob });
    } else if (out.role === 'PREDICTION') {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: pred });
    }
  }
}

// ── WeightedMedian ────────────────────────────────────────────────────────────

function executeWeightedMedian(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const attrs = node.attributes ?? [];
  const weightsAttr = attrs.find(a => a.name === 'weights');
  const weights: number[] = weightsAttr?.float64s ?? [];

  const pred = new Float64Array(N);

  for (let row = 0; row < N; row++) {
    // Collect (value, weight) pairs
    const pairs: { v: number; w: number }[] = [];
    for (let t = 0; t < inputNames.length; t++) {
      const td = namespace.get(inputNames[t]);
      if (!td) continue;
      const v = (td.data as Float64Array)[row];
      const w = weights[t] ?? 1;
      pairs.push({ v, w });
    }

    // Sort by value
    pairs.sort((a, b) => a.v - b.v);

    // Find weighted median: smallest index where cumulative weight >= totalWeight/2
    const totalWeight = pairs.reduce((s, p) => s + p.w, 0);
    let cumW = 0;
    let medianVal = pairs[0]?.v ?? 0;
    for (const { v, w } of pairs) {
      cumW += w;
      if (cumW >= totalWeight / 2) {
        medianVal = v;
        break;
      }
    }
    pred[row] = medianVal;
  }

  for (const out of node.outputs ?? []) {
    namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: pred });
  }
}

// ── Average ───────────────────────────────────────────────────────────────────

function executeAverage(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const inputs = inputNames.map(n => namespace.get(n)).filter(Boolean) as TensorData[];
  if (inputs.length === 0) return;
  const width = inputs[0].shape.length >= 2 ? (inputs[0].shape[inputs[0].shape.length - 1] ?? 1) : 1;
  const avg = new Float64Array(N * width);
  for (const td of inputs) {
    const d = td.data as Float64Array;
    for (let i = 0; i < avg.length; i++) avg[i] += d[i] ?? 0;
  }
  for (let i = 0; i < avg.length; i++) avg[i] /= inputs.length;
  const outputs = node.outputs ?? [];
  for (const out of outputs) {
    namespace.set(out.name, {
      dtype: 'FLOAT64',
      shape: width > 1 ? [N, width] : [N],
      data: avg,
    });
  }
}

// ── ArgMax ────────────────────────────────────────────────────────────────────

function executeArgMax(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const td = namespace.get(inputNames[0]);
  if (!td) return;
  const src = td.data as Float64Array;
  const cols = td.shape.length >= 2 ? td.shape[td.shape.length - 1] : 1;
  const out = new Float64Array(N);
  for (let row = 0; row < N; row++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let c = 0; c < cols; c++) {
      const v = src[row * cols + c];
      if (v > maxVal) { maxVal = v; maxIdx = c; }
    }
    out[row] = maxIdx;
  }
  const outName = node.outputs?.[0]?.name;
  if (outName) namespace.set(outName, { dtype: 'FLOAT64', shape: [N], data: out });
}

// ── SoftVote ──────────────────────────────────────────────────────────────────

function executeSoftVote(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const normalizeRows = node.attributes?.find(a => a.name === 'normalize_rows')?.b ?? false;

  const inputs = inputNames.map(n => namespace.get(n)).filter(Boolean) as TensorData[];
  if (inputs.length === 0) return;
  const colsPerInput = inputs.map(td => (td.shape.length >= 2 ? td.shape[td.shape.length - 1] : 1));
  const allSameWidth = colsPerInput.every(c => c === colsPerInput[0]);

  let prob: Float64Array;
  let cols: number;

  if (normalizeRows) {
    // OvR multiclass: stack single-column inputs into N×K, then normalize each row.
    cols = colsPerInput.reduce((a, b) => a + b, 0);
    const raw = new Float64Array(N * cols);
    let colOffset = 0;
    for (let ii = 0; ii < inputs.length; ii++) {
      const k = colsPerInput[ii];
      const src = inputs[ii].data as Float64Array;
      for (let row = 0; row < N; row++) {
        for (let c = 0; c < k; c++) raw[row * cols + colOffset + c] = src[row * k + c];
      }
      colOffset += k;
    }
    prob = new Float64Array(N * cols);
    for (let row = 0; row < N; row++) {
      let sum = 0;
      for (let k = 0; k < cols; k++) sum += raw[row * cols + k];
      for (let k = 0; k < cols; k++) {
        prob[row * cols + k] = sum > 0 ? raw[row * cols + k] / sum : 1 / cols;
      }
    }
  } else if (allSameWidth && inputs.length > 1) {
    // VotingClassifier soft: average per-class probabilities across all classifiers.
    cols = colsPerInput[0];
    prob = new Float64Array(N * cols);
    for (const td of inputs) {
      const src = td.data as Float64Array;
      for (let i = 0; i < N * cols; i++) prob[i] += src[i];
    }
    for (let i = 0; i < prob.length; i++) prob[i] /= inputs.length;
  } else {
    // Single input (or unequal widths): concatenate and pass through.
    cols = colsPerInput.reduce((a, b) => a + b, 0);
    prob = new Float64Array(N * cols);
    let colOffset = 0;
    for (let ii = 0; ii < inputs.length; ii++) {
      const k = colsPerInput[ii];
      const src = inputs[ii].data as Float64Array;
      for (let row = 0; row < N; row++) {
        for (let c = 0; c < k; c++) prob[row * cols + colOffset + c] = src[row * k + c];
      }
      colOffset += k;
    }
  }

  const pred = new Float64Array(N);
  for (let row = 0; row < N; row++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let k = 0; k < cols; k++) {
      if (prob[row * cols + k] > maxVal) { maxVal = prob[row * cols + k]; maxIdx = k; }
    }
    pred[row] = maxIdx;
  }

  for (const out of node.outputs ?? []) {
    if (out.role === 'PROBABILITY') {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N, cols], data: prob });
    } else if (out.role === 'PREDICTION') {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: pred });
    }
  }
}

// ── MajorityVote ──────────────────────────────────────────────────────────────

function executeMajorityVote(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): void {
  const pred = new Float64Array(N);
  for (let row = 0; row < N; row++) {
    const counts = new Map<number, number>();
    for (const name of inputNames) {
      const td = namespace.get(name);
      if (!td) continue;
      const label = Math.round((td.data as Float64Array)[row]);
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    let bestLabel = 0, bestCount = 0;
    for (const [label, count] of counts) {
      if (count > bestCount || (count === bestCount && label < bestLabel)) {
        bestCount = count; bestLabel = label;
      }
    }
    pred[row] = bestLabel;
  }

  for (const out of node.outputs ?? []) {
    namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: pred });
  }
}

// ── KNN ───────────────────────────────────────────────────────────────────────

function executeKNN(
  node: Node,
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
  resolved: ResolvedModel,
): void {
  const attrs = node.attributes ?? [];
  function attr(name: string) { return attrs.find(a => a.name === name); }
  function attrTd(name: string): TensorData | null {
    const a = attr(name);
    if (!a) return null;
    if (a.tensor) return tensorToData(a.tensor);
    if (a.tensor_ref) {
      const entry = resolved.tensorIndex.get(a.tensor_ref.id);
      if (entry?.dense) return tensorToData(entry.dense);
    }
    return null;
  }

  const trainFeatTd = attrTd('train_features');
  const trainTargTd = attrTd('train_targets');
  if (!trainFeatTd || !trainTargTd) return;

  const task        = attr('task')?.s ?? 'regression';
  const weights     = attr('weights')?.s ?? 'uniform';
  const neighborMode = attr('neighbor_mode')?.s ?? 'knn';
  const metric      = attr('metric')?.s ?? 'minkowski';
  const radius      = attr('radius')?.f64 ?? 1.0;
  const k           = attr('n_neighbors')?.i ?? 5;
  const nClasses    = attr('n_classes')?.i ?? 2;
  const outlierTd   = attrTd('outlier_label');
  const outlierLabel = outlierTd ? Number((outlierTd.data as Float64Array)[0]) : 0;

  // Minkowski p from metric_param_values (default p=2 = Euclidean).
  // Well-known named metrics override the param: manhattan=1, euclidean=2, chebyshev=Inf.
  const paramValsTd = attrTd('metric_param_values');
  const metricP = paramValsTd ? Number((paramValsTd.data as Float64Array)[0]) : 2;
  const p = metric === 'manhattan' ? 1
    : metric === 'euclidean' ? 2
    : metric === 'chebyshev' ? Infinity
    : metricP;

  const nTrain  = trainFeatTd.shape[0];
  const nFeat   = trainFeatTd.shape.length >= 2 ? trainFeatTd.shape[1] : 1;
  const trainX  = trainFeatTd.data as Float64Array;
  const trainY  = trainTargTd.data as Float64Array;

  // Merge query inputs into a single [N, nFeat] matrix
  const queryTd = namespace.get(inputNames[0]);
  if (!queryTd) return;
  const queryX = queryTd.data as Float64Array;

  function minkowski(ai: number, bi: number): number {
    if (p === Infinity) {
      let maxDist = 0;
      for (let f = 0; f < nFeat; f++) {
        const d = Math.abs(queryX[ai * nFeat + f] - trainX[bi * nFeat + f]);
        if (d > maxDist) maxDist = d;
      }
      return maxDist;
    }
    let sum = 0;
    for (let f = 0; f < nFeat; f++) {
      sum += Math.abs(queryX[ai * nFeat + f] - trainX[bi * nFeat + f]) ** p;
    }
    return sum ** (1 / p);
  }

  const isRadius    = neighborMode === 'radius';
  const isClassify  = task === 'classification';

  const predOut  = new Float64Array(N);
  const probOut  = isClassify ? new Float64Array(N * nClasses) : null;

  for (let row = 0; row < N; row++) {
    // Compute distances to all training points
    const dists: { d: number; idx: number }[] = [];
    for (let t = 0; t < nTrain; t++) {
      const d = minkowski(row, t);
      if (isRadius ? d <= radius : true) dists.push({ d, idx: t });
    }

    if (!isRadius) {
      // k-NN: keep only k nearest
      dists.sort((a, b) => a.d - b.d);
      dists.splice(k);
    }

    if (dists.length === 0) {
      // Outlier: no neighbors within radius
      predOut[row] = outlierLabel;
      if (probOut) {
        const cls = Math.round(outlierLabel);
        if (cls >= 0 && cls < nClasses) probOut[row * nClasses + cls] = 1;
        else probOut[row * nClasses] = 1;
      }
      continue;
    }

    if (isClassify) {
      const votes = new Float64Array(nClasses);
      for (const { d, idx } of dists) {
        const cls = Math.round(trainY[idx]);
        if (cls < 0 || cls >= nClasses) continue;
        const w = (weights === 'distance') ? (d === 0 ? 1e12 : 1 / d) : 1;
        votes[cls] += w;
      }
      const total = votes.reduce((a, b) => a + b, 0);
      let maxV = -Infinity, maxC = 0;
      for (let c = 0; c < nClasses; c++) {
        if (probOut) probOut[row * nClasses + c] = total > 0 ? votes[c] / total : 1 / nClasses;
        if (votes[c] > maxV) { maxV = votes[c]; maxC = c; }
      }
      predOut[row] = maxC;
    } else {
      let num = 0, den = 0;
      for (const { d, idx } of dists) {
        const w = (weights === 'distance') ? (d === 0 ? 1e12 : 1 / d) : 1;
        num += w * trainY[idx];
        den += w;
      }
      predOut[row] = den > 0 ? num / den : 0;
    }
  }

  for (const out of node.outputs ?? []) {
    if (out.role === 'PROBABILITY' && probOut) {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N, nClasses], data: probOut });
    } else if (out.role === 'PREDICTION' || out.role === 'SCORE' || !out.role) {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: predOut });
    }
  }
}

// ── Slot space builder ────────────────────────────────────────────────────────

function buildFlatSlotSpace(
  inputNames: string[],
  namespace: Map<string, TensorData>,
  N: number,
): { flatInputs: Float64Array; numSlots: number } {
  // Count total slots
  let numSlots = 0;
  for (const name of inputNames) {
    const td = namespace.get(name);
    numSlots += td ? tensorCols(td) : 0;
  }

  const flatInputs = new Float64Array(N * numSlots);
  let slotOffset = 0;

  for (const name of inputNames) {
    const td = namespace.get(name);
    if (!td) continue;
    const cols = tensorCols(td);
    const data = td.data as Float64Array;
    for (let row = 0; row < N; row++) {
      for (let col = 0; col < cols; col++) {
        flatInputs[row * numSlots + slotOffset + col] = data[row * cols + col];
      }
    }
    slotOffset += cols;
  }

  return { flatInputs, numSlots };
}

// ── Output publishing ─────────────────────────────────────────────────────────

function publishNumericOutputs(
  node: Node,
  outputData: Float64Array,
  outputWidth: number,
  N: number,
  namespace: Map<string, TensorData>,
) {
  const outputs = node.outputs ?? [];

  if (outputs.length === 1) {
    // Expand binary scalar probability to [N, 2] so downstream TakeSlots/ArgMax work correctly.
    if (outputs[0].role === 'PROBABILITY' && outputWidth === 1) {
      const expanded = new Float64Array(N * 2);
      for (let row = 0; row < N; row++) {
        expanded[row * 2] = 1 - outputData[row];
        expanded[row * 2 + 1] = outputData[row];
      }
      namespace.set(outputs[0].name, { dtype: 'FLOAT64', shape: [N, 2], data: expanded });
      return;
    }
    namespace.set(outputs[0].name, {
      dtype: 'FLOAT64',
      shape: outputWidth > 1 ? [N, outputWidth] : [N],
      data: outputData,
    });
    return;
  }

  // When a PROBABILITY output is present, outputData contains post-transform
  // probabilities. PREDICTION must be derived (argmax or 0.5 threshold).
  const hasProbOutput = outputs.some(o => o.role === 'PROBABILITY');

  // Binary classification: scalar probability → expand to [1-p, p] two-column output
  const isBinaryScalar = hasProbOutput && outputWidth === 1;
  let probData = outputData;
  let probWidth = outputWidth;
  if (isBinaryScalar) {
    probData = new Float64Array(N * 2);
    for (let row = 0; row < N; row++) {
      probData[row * 2] = 1 - outputData[row];
      probData[row * 2 + 1] = outputData[row];
    }
    probWidth = 2;
  }

  for (const out of outputs) {
    if (out.role === 'PREDICTION' && hasProbOutput) {
      const pred = new Float64Array(N);
      for (let row = 0; row < N; row++) {
        let maxVal = -Infinity, maxIdx = 0;
        for (let k = 0; k < probWidth; k++) {
          const v = probData[row * probWidth + k];
          if (v > maxVal) { maxVal = v; maxIdx = k; }
        }
        pred[row] = maxIdx;
      }
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: pred });
    } else if (out.role === 'PROBABILITY') {
      namespace.set(out.name, {
        dtype: 'FLOAT64',
        shape: [N, probWidth],
        data: probData,
      });
    } else {
      // SCORE / CONFIDENCE / no role: pass raw output as-is
      namespace.set(out.name, {
        dtype: 'FLOAT64',
        shape: outputWidth > 1 ? [N, outputWidth] : [N],
        data: outputData,
      });
    }
  }
}

function publishClusteringOutputs(
  node: Node,
  labels: Int32Array,
  distances: Float64Array,
  namespace: Map<string, TensorData>,
  N: number,
) {
  const outputs = node.outputs ?? [];
  for (const out of outputs) {
    const role = out.role;
    if (role === 'PREDICTION' || outputs.indexOf(out) === 0) {
      const data = new Float64Array(labels.length);
      for (let i = 0; i < labels.length; i++) data[i] = labels[i];
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data });
    } else {
      namespace.set(out.name, { dtype: 'FLOAT64', shape: [N], data: distances });
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function serializeTensor(td: TensorData | undefined, max = 50): SerializedTensor {
  if (!td) return { dtype: 'FLOAT64', shape: [], data: null };
  const raw = td.data;
  let data: (number | string | boolean)[];
  if (raw instanceof Float64Array) {
    data = Array.from(raw).slice(0, max) as number[];
  } else {
    data = (raw as (string | boolean)[]).slice(0, max);
  }
  return { dtype: td.dtype, shape: td.shape, data };
}

function inferBatchSize(namespace: Map<string, TensorData>): number {
  for (const td of namespace.values()) {
    if (td.shape.length > 0 && td.shape[0] > 0) return td.shape[0];
  }
  return 1;
}

export { resolve };
