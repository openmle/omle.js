// ──────────────────────────────────────────────────────────────────────────────
// AUTO-GENERATED — do not edit by hand.
// Source: github.com/openmle/omle @ main — protobuf/omle.proto
// Regenerate: npm run gen:ir           (OMLE_PROTO_REF=<tag> to pin a release)
// ──────────────────────────────────────────────────────────────────────────────
//
// OMLE v1 TypeScript IR — mirrors omle.proto field-for-field.
// JSON serialisation convention (from the Python reference SDK):
//   - enums are string names ("FLOAT32", not 11)
//   - absent / empty fields are omitted
//   - bytes fields are base64-encoded strings
//   - typed tensor payload lists (float32_data etc.) are flat arrays, NOT {values:[]}
//   - int64 values are represented as number (safe up to 2^53)

// ── Enums ──────────────────────────────────────────────────────────────────────

export type DataType =
  | 'DATA_TYPE_UNSPECIFIED' | 'BOOL' | 'INT8' | 'INT16' | 'INT32' | 'INT64' | 'UINT8' | 'UINT16' | 'UINT32' | 'UINT64' | 'FLOAT16' | 'FLOAT32' | 'FLOAT64' | 'STRING' | 'BYTES' | 'DATE' | 'TIME' | 'TIMESTAMP';

export type MeasureLevel =
  | 'MEASURE_LEVEL_UNSPECIFIED' | 'CONTINUOUS' | 'NOMINAL' | 'ORDINAL' | 'FLAG';

export type TaskType =
  | 'TASK_TYPE_UNSPECIFIED' | 'REGRESSION' | 'BINARY' | 'MULTICLASS' | 'CLUSTERING' | 'ANOMALY_DETECTION';

export type OutputRole =
  | 'OUTPUT_ROLE_UNSPECIFIED' | 'PREDICTION' | 'PROBABILITY' | 'SCORE' | 'CONFIDENCE' | 'STANDARD_ERROR' | 'STANDARD_DEVIATION' | 'RESIDUAL' | 'TRANSFORMED_VALUE' | 'ENTITY_ID' | 'AFFINITY' | 'CONTRIBUTION' | 'INTERMEDIATE';

export type PostTransform =
  | 'POST_TRANSFORM_UNSPECIFIED' | 'IDENTITY' | 'SIGMOID' | 'SIGMOID_BINARY' | 'SOFTMAX' | 'LOGIT' | 'PROBIT' | 'EXP' | 'CLOGLOG' | 'CAUCHIT' | 'LOGLOG';

export type MissingValuePolicy =
  | 'MISSING_PROPAGATE' | 'MISSING_AS_VALUE' | 'MISSING_AS_INVALID';

export type InvalidValuePolicy =
  | 'INVALID_RETURN_INVALID' | 'INVALID_AS_IS' | 'INVALID_AS_MISSING' | 'INVALID_AS_VALUE';

export type OutlierValuePolicy =
  | 'OUTLIER_AS_IS' | 'OUTLIER_AS_MISSING' | 'OUTLIER_AS_EXTREME';

export type TargetKind =
  | 'TARGET_KIND_UNSPECIFIED' | 'REGRESSION' | 'BINARY' | 'MULTICLASS';

export type IntervalClosure =
  | 'CLOSURE_UNSPECIFIED' | 'OPEN_OPEN' | 'OPEN_CLOSED' | 'CLOSED_OPEN' | 'CLOSED_CLOSED';

export type DomainValueProperty =
  | 'VALID' | 'INVALID' | 'MISSING';

export type SimplePredicateOp =
  | 'OPERATOR_UNSPECIFIED' | 'LESS_THAN' | 'LESS_OR_EQUAL' | 'GREATER_THAN' | 'GREATER_OR_EQUAL' | 'EQUAL' | 'NOT_EQUAL' | 'IS_MISSING' | 'IS_NOT_MISSING';

export type SimpleSetPredicateOp =
  | 'OPERATOR_UNSPECIFIED' | 'IN' | 'NOT_IN';

export type CompoundBooleanOp =
  | 'BOOLEAN_OPERATOR_UNSPECIFIED' | 'AND' | 'OR' | 'XOR' | 'SURROGATE';

export type TreeNodeKind =
  | 'NODE_KIND_UNSPECIFIED' | 'LEAF' | 'BRANCH';

export type TreeSplitOp =
  | 'SPLIT_OP_UNSPECIFIED' | 'LESS_THAN' | 'LESS_OR_EQUAL' | 'GREATER_THAN' | 'GREATER_OR_EQUAL' | 'EQUAL' | 'NOT_EQUAL' | 'IN_SET' | 'NOT_IN_SET' | 'IS_MISSING';

export type TreeEnsembleAggregation =
  | 'AGGREGATION_UNSPECIFIED' | 'SUM' | 'AVERAGE' | 'WEIGHTED_SUM' | 'WEIGHTED_AVERAGE' | 'MAJORITY_VOTE' | 'SOFT_VOTE' | 'MIN' | 'MAX';

export type DistanceMeasure =
  | 'DISTANCE_MEASURE_UNSPECIFIED' | 'EUCLIDEAN' | 'SQUARED_EUCLIDEAN' | 'MANHATTAN' | 'COSINE';

export type CovarianceType =
  | 'COVARIANCE_TYPE_UNSPECIFIED' | 'FULL' | 'DIAGONAL' | 'SPHERICAL';

export type SVMMulticlassStrategy =
  | 'MULTICLASS_STRATEGY_UNSPECIFIED' | 'ONE_VS_REST' | 'ONE_VS_ONE';

export type KernelType =
  | 'KERNEL_TYPE_UNSPECIFIED' | 'LINEAR' | 'POLY' | 'RBF' | 'SIGMOID';

export type NNActivation =
  | 'ACTIVATION_UNSPECIFIED' | 'IDENTITY' | 'LOGISTIC' | 'TANH' | 'RELU' | 'SOFTMAX';

export type DetectionMode =
  | 'DETECTION_MODE_UNSPECIFIED' | 'OUTLIER_DETECTION' | 'NOVELTY_DETECTION';

export type ScorePolarity =
  | 'SCORE_POLARITY_UNSPECIFIED' | 'HIGHER_MORE_ABNORMAL' | 'LOWER_MORE_ABNORMAL';

// ── Messages ───────────────────────────────────────────────────────────────────

export interface OMLEModel {
  metadata?: ModelMetadata;
  operator_imports?: NamespaceImport[];
  function_imports?: NamespaceImport[];
  inputs?: InputSpec[];
  outputs?: OutputSpec[];
  model_schema?: ModelSchema;
  nodes?: Node[];
  verification?: ModelVerification;
  warmup?: RuntimeWarmup;
  sample_inputs?: SampleInputSet;
  tensor_entries?: TensorEntry[];
  functions?: DefineFunction[];
}

export interface SourceFramework {
  name?: string;
  version?: string;
  role?: string;
}

export interface ModelMetadata {
  format_version?: string;
  name?: string;
  version?: string;
  timestamp?: string;
  producer?: string;
  producer_version?: string;
  source_frameworks?: SourceFramework[];
  doc_string?: string;
  copyright?: string;
  attributes?: Record<string, string>;
}

export interface NamespaceImport {
  namespace: string;
  version?: string;
}

export interface Scalar {
  // value oneof
  bool_value?: boolean;
  int_value?: number;
  float_value?: number;
  double_value?: number;
  string_value?: string;
}

export interface InputSpec {
  name: string;
  type?: TensorType;
  description?: string;
  attributes?: Record<string, string>;
}

export interface OutputSpec {
  name: string;
  type?: TensorType;
  role?: OutputRole;
  binding?: OutputBinding;
  description?: string;
  field_names?: string[];
  attributes?: Record<string, string>;
}

export interface OutputBinding {
  target_name?: string;
  values?: Scalar[];
  segment_id?: string;
  rank?: number;
  attributes?: Record<string, string>;
}

export interface ModelSchema {
  features?: Feature[];
  targets?: Target[];
}

export interface NameRange {
  prefix: string;
  start: number;
  end: number;
  width?: number;
}

export interface Feature {
  // naming oneof
  name?: string;
  range?: NameRange;
  description?: string;
  type?: TensorType;
  measure_level?: MeasureLevel;
  source?: string;
  index?: number;
  missing_value_policy?: MissingValuePolicy;
  missing_replacement_value?: Scalar;
  invalid_value_policy?: InvalidValuePolicy;
  invalid_replacement_value?: Scalar;
  outlier_value_policy?: OutlierValuePolicy;
  domain?: ValueDomain;
  attributes?: Record<string, string>;
}

export interface Target {
  name: string;
  type?: TensorType;
  kind?: TargetKind;
  measure_level?: MeasureLevel;
  class_labels?: Scalar[];
  description?: string;
  attributes?: Record<string, string>;
}

export interface ValueDomain {
  continuous?: ContinuousDomain;
  discrete?: DiscreteDomain;
}

export interface ContinuousDomain {
  intervals?: Interval[];
}

export interface Interval {
  left_margin: Scalar;
  right_margin: Scalar;
  closure: IntervalClosure;
}

export interface DiscreteDomain {
  values?: DomainValue[];
  ordered?: boolean;
}

export interface DomainValue {
  value?: Scalar;
  original_value?: Scalar;
  display_name?: string;
  property?: DomainValueProperty;
}

export interface TensorType {
  dtype: DataType;
  shape: number[];
}

export interface Tensor {
  name?: string;
  type?: TensorType;
  // data oneof
  raw_data?: string;
  float32_data?: number[];
  float64_data?: number[];
  int32_data?: number[];
  int64_data?: number[];
  string_data?: string[];
  bytes_data?: string[];
  bool_data?: boolean[];
}

export interface SparseTensor {
  name?: string;
  type?: TensorType;
  default_value?: Scalar;
  csr?: CSRMatrix;
}

export interface CSRMatrix {
  indices: number[];
  indptr: number[];
  // data oneof
  raw_data?: string;
  float32_data?: number[];
  float64_data?: number[];
  int32_data?: number[];
  int64_data?: number[];
  string_data?: string[];
}

export interface TensorEntry {
  id: string;
  // value oneof
  dense?: Tensor;
  sparse?: SparseTensor;
  attributes?: Record<string, string>;
}

export interface TensorRef {
  id: string;
}

export interface TensorValue {
  // value oneof
  tensor?: Tensor;
  sparse?: SparseTensor;
  tensor_ref?: TensorRef;
}

export interface NameRef {
  value: string;
  field?: string;
}

export interface Expression {
  // kind oneof
  literal?: Scalar;
  ref?: NameRef;
  apply?: Apply;
}

export interface Apply {
  function: string;
  arguments?: Expression[];
}

export interface Predicate {
  // kind oneof
  true_predicate?: TruePredicate;
  false_predicate?: FalsePredicate;
  simple?: SimplePredicate;
  simple_set?: SimpleSetPredicate;
  compound?: CompoundPredicate;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface TruePredicate {}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface FalsePredicate {}

export interface SimplePredicate {
  column: NameRef;
  op: SimplePredicateOp;
  value?: Scalar;
}

export interface SimpleSetPredicate {
  column: NameRef;
  op: SimpleSetPredicateOp;
  values?: Scalar[];
}

export interface CompoundPredicate {
  op: CompoundBooleanOp;
  predicates?: Predicate[];
}

export interface DefineFunction {
  name: string;
  doc_string?: string;
  parameters?: FunctionParameter[];
  result_data_type?: DataType;
  result_measure_level?: MeasureLevel;
  body?: Expression;
  attributes?: Record<string, string>;
}

export interface NodeInput {
  // ref oneof
  name?: NameRef;
  range?: NameRange;
}

export interface NodeOutput {
  name: string;
  type?: TensorType;
  measure_level?: MeasureLevel;
  domain?: ValueDomain;
  description?: string;
  role?: OutputRole;
  binding?: OutputBinding;
  field_names?: string[];
  attributes?: Record<string, string>;
}

export interface NameAlias {
  from_name: string;
  to_name: string;
}

export interface CompositeNode {
  input_aliases?: NameAlias[];
  nodes?: Node[];
  output_aliases?: NameAlias[];
}

export interface Node {
  name: string;
  domain?: string;
  op?: string;
  inputs?: NodeInput[];
  outputs?: NodeOutput[];
  attributes?: Attribute[];
  // body oneof
  composite?: CompositeNode;
  tree?: Tree;
  tree_ensemble?: TreeEnsemble;
  linear?: Linear;
  naive_bayes?: NaiveBayes;
  clustering?: Clustering;
  svm?: SVM;
  neural_network?: NeuralNetwork;
  anomaly_detection?: AnomalyDetection;
  metadata?: Record<string, string>;
}

export interface Attribute {
  name: string;
  // value oneof
  i?: number;
  f32?: number;
  f64?: number;
  s?: string;
  b?: boolean;
  ints?: number[];
  float32s?: number[];
  float64s?: number[];
  strings?: string[];
  bools?: boolean[];
  tensor_ref?: TensorRef;
  tensor?: Tensor;
  sparse?: SparseTensor;
  type?: TensorType;
  expr?: Expression;
  predicate?: Predicate;
}

export interface Tree {
  task_type?: TaskType;
  num_nodes?: number;
  node_kind?: TreeNodeKind[];
  split_feature?: number[];
  split_threshold?: TensorValue;
  split_op?: TreeSplitOp[];
  category_set_offset?: number[];
  category_set_count?: number[];
  category_set?: number[];
  complex_predicates?: ComplexPredicate[];
  children_index?: number[];
  children_offset?: number[];
  children_count?: number[];
  default_child?: number[];
  leaf_value?: TensorValue;
  leaf_width?: number;
  leaf_vector?: TensorValue;
  leaf_vector_index?: number[];
}

export interface TreeEnsemble {
  task_type?: TaskType;
  trees?: Tree[];
  aggregation?: TreeEnsembleAggregation;
  // _tree_weights oneof
  tree_weights?: TensorValue;
  // _base_score oneof
  base_score?: Scalar;
  tree_group?: number[];
  post_transform?: PostTransform;
}

export interface Linear {
  task_type?: TaskType;
  coefficients: TensorValue;
  // _intercept oneof
  intercept?: TensorValue;
  // _weight_covariances oneof
  weight_covariances?: TensorValue;
  // _noise_precision oneof
  noise_precision?: TensorValue;
  post_transform?: PostTransform;
}

export interface NaiveBayes {
  task_type?: TaskType;
  class_log_priors: TensorValue;
  // implementation oneof
  gaussian?: GaussianNaiveBayes;
  multinomial?: MultinomialNaiveBayes;
  bernoulli?: BernoulliNaiveBayes;
  categorical?: CategoricalNaiveBayes;
}

export interface GaussianNaiveBayes {
  means: TensorValue;
  variances: TensorValue;
  // _variance_epsilon oneof
  variance_epsilon?: Scalar;
}

export interface MultinomialNaiveBayes {
  feature_log_prob: TensorValue;
}

export interface BernoulliNaiveBayes {
  feature_log_prob: TensorValue;
  // _binarize_threshold oneof
  binarize_threshold?: Scalar;
}

export interface CategoricalNaiveBayes {
  category_log_prob: TensorValue;
  category_offset?: number[];
  category_count?: number[];
}

export interface Clustering {
  task_type?: TaskType;
  // implementation oneof
  prototype?: PrototypeClustering;
  gaussian_mixture?: GaussianMixtureClustering;
}

export interface PrototypeClustering {
  centers: TensorValue;
  distance_measure?: DistanceMeasure;
  cluster_labels?: Scalar[];
}

export interface GaussianMixtureClustering {
  weights: TensorValue;
  means: TensorValue;
  covariances: TensorValue;
  covariance_type?: CovarianceType;
  component_labels?: Scalar[];
}

export interface SVM {
  task_type?: TaskType;
  // _multiclass_strategy oneof
  multiclass_strategy?: SVMMulticlassStrategy;
  // implementation oneof
  linear?: LinearSVM;
  kernel?: KernelSVM;
  post_transform?: PostTransform;
}

export interface LinearSVM {
  coefficients: TensorValue;
  // _intercept oneof
  intercept?: TensorValue;
}

export interface KernelSVM {
  kernel_type?: KernelType;
  support_vectors: TensorValue;
  dual_coefficients: TensorValue;
  // _intercept oneof
  intercept?: TensorValue;
  // _gamma oneof
  gamma?: Scalar;
  // _degree oneof
  degree?: number;
  // _coef0 oneof
  coef0?: Scalar;
  n_support?: number[];
  // _prob_a oneof
  prob_a?: TensorValue;
  // _prob_b oneof
  prob_b?: TensorValue;
}

export interface NeuralNetwork {
  task_type?: TaskType;
  layers?: DenseLayer[];
}

export interface AnomalyDetection {
  task_type?: TaskType;
  mode?: DetectionMode;
  raw_score_polarity?: ScorePolarity;
  // _threshold oneof
  threshold?: Scalar;
  // implementation oneof
  isolation_forest?: IsolationForest;
  one_class_svm?: OneClassSVM;
  linear_one_class_svm?: LinearOneClassSVM;
  local_outlier_factor?: LocalOutlierFactor;
  elliptic_envelope?: EllipticEnvelope;
}

export interface IsolationForest {
  trees?: Tree[];
  max_samples?: number;
  // _offset oneof
  offset?: Scalar;
}

export interface OneClassSVM {
  kernel_svm?: KernelSVM;
  // _offset oneof
  offset?: Scalar;
}

export interface LinearOneClassSVM {
  coefficients?: TensorValue;
  // _intercept oneof
  intercept?: TensorValue;
  // _offset oneof
  offset?: Scalar;
}

export interface LocalOutlierFactor {
  reference_samples?: TensorValue;
  n_neighbors?: number;
  metric?: string;
  metric_params?: Record<string, string>;
  // _offset oneof
  offset?: Scalar;
}

export interface EllipticEnvelope {
  location?: TensorValue;
  covariance?: TensorValue;
  // _precision oneof
  precision?: TensorValue;
  // _offset oneof
  offset?: Scalar;
}

export interface NumericTolerance {
  // _atol oneof
  atol?: Scalar;
  // _rtol oneof
  rtol?: Scalar;
}

export interface ModelVerification {
  cases?: VerificationCase[];
  tolerance?: NumericTolerance;
}

export interface VerificationCase {
  inputs?: TensorRef[];
  expected_outputs?: TensorRef[];
  description?: string;
}

export interface RuntimeWarmup {
  cases?: WarmupCase[];
}

export interface WarmupCase {
  inputs?: TensorRef[];
  description?: string;
  repeat?: number;
}

export interface SampleInputSet {
  cases?: SampleInputCase[];
}

export interface SampleInputCase {
  inputs?: TensorRef[];
  description?: string;
}

export interface FunctionParameter {
  name: string;
  data_type?: DataType;
  measure_level?: MeasureLevel;
}

export interface ComplexPredicate {
  node_index: number;
  predicate: Predicate;
}

export interface DenseLayer {
  weights: TensorValue;
  // _bias oneof
  bias?: TensorValue;
  activation: NNActivation;
  name?: string;
}


// ── Runtime helpers (not auto-generated) ─────────────────────────────────────

export function scalarValue(s: Scalar | null | undefined): boolean | number | string | undefined {
  if (s == null) return undefined;
  if (s.bool_value !== undefined) return s.bool_value;
  if (s.int_value !== undefined) return s.int_value;
  if (s.float_value !== undefined) return s.float_value;
  if (s.double_value !== undefined) return s.double_value;
  return s.string_value;
}

export function scalarToNumber(s: Scalar | null | undefined): number {
  if (s == null) return NaN;
  if (s.double_value !== undefined) return s.double_value;
  if (s.float_value !== undefined) return s.float_value;
  if (s.int_value !== undefined) return s.int_value;
  if (s.bool_value !== undefined) return s.bool_value ? 1 : 0;
  return NaN;
}
