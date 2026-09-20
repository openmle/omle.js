// Validation for OMLE models.
//
// Produces a ValidationResult with errors and warnings.
// Does not throw — callers can decide how to handle issues.

import type { OMLEModel, Node, TensorRef, TensorValue, DefineFunction } from './ir.js';
import { expandNodeInputs, expandFeatures } from './resolve.js';

// ── Public types ──────────────────────────────────────────────────────────────

export interface ValidationIssue {
  severity: 'error' | 'warning';
  path: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  issues: ValidationIssue[];
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
}

// ── Main entry point ──────────────────────────────────────────────────────────

export function validate(model: OMLEModel): ValidationResult {
  const issues: ValidationIssue[] = [];

  checkMetadata(model, issues);
  checkInputsOutputs(model, issues);
  checkTensorEntries(model, issues);
  checkModelSchema(model, issues);
  checkNodes(model, issues);
  checkFunctions(model, issues);
  checkOutputResolution(model, issues);

  const errors = issues.filter(i => i.severity === 'error');
  const warnings = issues.filter(i => i.severity === 'warning');
  return { valid: errors.length === 0, issues, errors, warnings };
}

export function assertValid(model: OMLEModel): void {
  const result = validate(model);
  if (!result.valid) {
    const msgs = result.errors.map(e => `  [${e.path}] ${e.message}`).join('\n');
    throw new Error(`OMLE model is invalid:\n${msgs}`);
  }
}

// ── Checker helpers ───────────────────────────────────────────────────────────

function err(issues: ValidationIssue[], path: string, message: string) {
  issues.push({ severity: 'error', path, message });
}

function warn(issues: ValidationIssue[], path: string, message: string) {
  issues.push({ severity: 'warning', path, message });
}

// The OMLE format is pre-1.0 — omle.proto states "The format is currently
// pre-1.0" and gives "0.1.0" as its example — so 0.x is what every model in the
// wild carries. 1.x is accepted ahead of the v1 release.
const SUPPORTED_MAJORS = new Set([0, 1]);

function checkMetadata(model: OMLEModel, issues: ValidationIssue[]) {
  const ver = model.metadata?.format_version;
  if (!ver) {
    // An error, matching omle/validation.py — the two implementations must
    // agree on whether a given file is valid.
    err(issues, 'metadata.format_version', 'format_version must not be empty');
    return;
  }
  const [major] = ver.split('.').map(Number);
  if (!Number.isFinite(major)) {
    err(issues, 'metadata.format_version', `Malformed format_version: ${ver}`);
    return;
  }
  // A warning, not an error: the Python SDK loads any non-empty version, and
  // erroring here would make the two implementations disagree about one file.
  if (!SUPPORTED_MAJORS.has(major)) {
    warn(
      issues,
      'metadata.format_version',
      `Unrecognized major version: ${major} (this build understands 0.x and 1.x)`,
    );
  }
}

function checkInputsOutputs(model: OMLEModel, issues: ValidationIssue[]) {
  const inputNames = new Set<string>();
  for (const [i, inp] of (model.inputs ?? []).entries()) {
    if (!inp.name) {
      err(issues, `inputs[${i}].name`, 'Input name is required');
    } else if (inputNames.has(inp.name)) {
      err(issues, `inputs[${i}].name`, `Duplicate input name: ${inp.name}`);
    } else {
      inputNames.add(inp.name);
    }
  }

  const outputNames = new Set<string>();
  for (const [i, out] of (model.outputs ?? []).entries()) {
    if (!out.name) {
      err(issues, `outputs[${i}].name`, 'Output name is required');
    } else if (outputNames.has(out.name)) {
      err(issues, `outputs[${i}].name`, `Duplicate output name: ${out.name}`);
    } else {
      outputNames.add(out.name);
    }
  }
}

function checkTensorEntries(model: OMLEModel, issues: ValidationIssue[]) {
  const ids = new Set<string>();
  for (const [i, entry] of (model.tensor_entries ?? []).entries()) {
    if (!entry.id) {
      err(issues, `tensor_entries[${i}].id`, 'TensorEntry id is required');
    } else if (ids.has(entry.id)) {
      err(issues, `tensor_entries[${i}].id`, `Duplicate tensor entry id: ${entry.id}`);
    } else {
      ids.add(entry.id);
    }
    if (!entry.dense && !entry.sparse) {
      warn(issues, `tensor_entries[${i}]`, 'TensorEntry has neither dense nor sparse value');
    }
  }
}

function checkModelSchema(model: OMLEModel, issues: ValidationIssue[]) {
  const schema = model.model_schema;
  if (!schema) return;

  const inputNames = new Set((model.inputs ?? []).map(i => i.name));
  const expanded = expandFeatures(schema.features ?? []);
  const featureNames = new Set<string>();

  for (const [i, f] of expanded.entries()) {
    const name = f.name ?? `features[${i}]`;
    if (!f.name) {
      err(issues, `model_schema.features[${i}]`, 'Feature must have a name or range');
    } else if (featureNames.has(f.name)) {
      err(issues, `model_schema.features[${i}].name`, `Duplicate feature name: ${f.name}`);
    } else {
      featureNames.add(f.name);
    }
    const source = f.source ?? f.name;
    if (source && !inputNames.has(source)) {
      warn(issues, `model_schema.features[${i}].source`,
        `Feature "${name}" source "${source}" does not match any InputSpec`);
    }
  }

  const targetNames = new Set<string>();
  for (const [i, t] of (schema.targets ?? []).entries()) {
    if (!t.name) {
      err(issues, `model_schema.targets[${i}].name`, 'Target name is required');
    } else if (targetNames.has(t.name)) {
      err(issues, `model_schema.targets[${i}].name`, `Duplicate target name: ${t.name}`);
    } else {
      targetNames.add(t.name);
    }
  }
}

function checkNodes(model: OMLEModel, issues: ValidationIssue[]) {
  const tensorIds = new Set((model.tensor_entries ?? []).map(e => e.id));
  const available = new Set((model.inputs ?? []).map(i => i.name));

  // Add schema feature names
  const schema = model.model_schema;
  if (schema?.features) {
    for (const f of expandFeatures(schema.features)) {
      if (f.name) available.add(f.name);
    }
  }

  const nodeNames = new Set<string>();
  for (const [i, node] of (model.nodes ?? []).entries()) {
    const path = `nodes[${i}]`;
    if (!node.name) {
      err(issues, `${path}.name`, 'Node name is required');
    } else if (nodeNames.has(node.name)) {
      err(issues, `${path}.name`, `Duplicate node name: ${node.name}`);
    } else {
      nodeNames.add(node.name);
    }

    // Check inputs resolve
    const inputNames = expandNodeInputs(node.inputs ?? []);
    for (const name of inputNames) {
      if (!available.has(name)) {
        err(issues, `${path}.inputs`, `Input "${name}" is not available in scope`);
      }
    }

    // Check output name uniqueness
    const outNames = new Set<string>();
    for (const out of node.outputs ?? []) {
      if (!out.name) {
        err(issues, `${path}.outputs`, 'NodeOutput.name is required');
      } else if (outNames.has(out.name)) {
        err(issues, `${path}.outputs`, `Duplicate output name: ${out.name}`);
      } else {
        outNames.add(out.name);
        available.add(out.name);
      }
    }

    // Check TensorRefs in body
    checkNodeBodyRefs(node, path, tensorIds, issues);
  }
}

function checkNodeBodyRefs(
  node: Node,
  path: string,
  tensorIds: Set<string>,
  issues: ValidationIssue[],
) {
  function checkRef(tv: TensorValue | TensorRef | null | undefined, refPath: string) {
    if (!tv) return;
    // TensorValue: check tensor_ref if present; inline tensor needs no lookup
    if ('tensor_ref' in tv || 'tensor' in tv || 'sparse' in tv) {
      const ref = (tv as TensorValue).tensor_ref;
      if (ref && !tensorIds.has(ref.id)) {
        err(issues, refPath, `TensorRef "${ref.id}" not found in tensor_entries`);
      }
    } else {
      // Legacy TensorRef
      const ref = tv as TensorRef;
      if (!tensorIds.has(ref.id)) {
        err(issues, refPath, `TensorRef "${ref.id}" not found in tensor_entries`);
      }
    }
  }

  const b = node;
  if (b.linear) {
    checkRef(b.linear.coefficients, `${path}.linear.coefficients`);
  }
  if (b.tree_ensemble) {
    // no direct refs in tree ensemble body (trees are embedded)
  }
  if (b.neural_network) {
    for (const [i, layer] of (b.neural_network.layers ?? []).entries()) {
      checkRef(layer.weights, `${path}.neural_network.layers[${i}].weights`);
      // layer.bias is an inline Tensor, not a TensorRef — no ref check needed
    }
  }
  if (b.naive_bayes) {
    checkRef(b.naive_bayes.class_log_priors, `${path}.naive_bayes.class_log_priors`);
    const nb = b.naive_bayes;
    if (nb.gaussian) {
      checkRef(nb.gaussian.means, `${path}.naive_bayes.gaussian.means`);
      checkRef(nb.gaussian.variances, `${path}.naive_bayes.gaussian.variances`);
    }
    if (nb.multinomial) checkRef(nb.multinomial.feature_log_prob, `${path}.naive_bayes.multinomial.feature_log_prob`);
    if (nb.bernoulli) checkRef(nb.bernoulli.feature_log_prob, `${path}.naive_bayes.bernoulli.feature_log_prob`);
    if (nb.categorical) checkRef(nb.categorical.category_log_prob, `${path}.naive_bayes.categorical.category_log_prob`);
  }
  if (b.clustering) {
    if (b.clustering.prototype) checkRef(b.clustering.prototype.centers, `${path}.clustering.prototype.centers`);
    if (b.clustering.gaussian_mixture) {
      const gm = b.clustering.gaussian_mixture;
      checkRef(gm.weights, `${path}.clustering.gaussian_mixture.weights`);
      checkRef(gm.means, `${path}.clustering.gaussian_mixture.means`);
      checkRef(gm.covariances, `${path}.clustering.gaussian_mixture.covariances`);
    }
  }
  if (b.svm) {
    if (b.svm.linear) {
      checkRef(b.svm.linear.coefficients, `${path}.svm.linear.coefficients`);
    }
    if (b.svm.kernel) {
      checkRef(b.svm.kernel.support_vectors, `${path}.svm.kernel.support_vectors`);
      checkRef(b.svm.kernel.dual_coefficients, `${path}.svm.kernel.dual_coefficients`);
    }
  }
}

function checkFunctions(model: OMLEModel, issues: ValidationIssue[]) {
  const names = new Set<string>();
  for (const [i, fn] of (model.functions ?? []).entries()) {
    if (!fn.name) {
      err(issues, `functions[${i}].name`, 'DefineFunction name is required');
    } else if (names.has(fn.name)) {
      err(issues, `functions[${i}].name`, `Duplicate function name: ${fn.name}`);
    } else {
      names.add(fn.name);
    }
    // Check for duplicate param names
    const paramNames = new Set<string>();
    for (const [j, p] of (fn.parameters ?? []).entries()) {
      if (paramNames.has(p.name)) {
        err(issues, `functions[${i}].parameters[${j}].name`,
          `Duplicate parameter name: ${p.name}`);
      }
      paramNames.add(p.name);
    }
  }
  // Detect function call cycles (simple DFS)
  checkFunctionCycles(model.functions ?? [], issues);
}

function checkFunctionCycles(fns: DefineFunction[], issues: ValidationIssue[]) {
  const fnNames = new Set(fns.map(f => f.name));
  const callGraph = new Map<string, Set<string>>();

  function collectCalls(expr: unknown, callee: string) {
    if (!expr || typeof expr !== 'object') return;
    const e = expr as Record<string, unknown>;
    if (e['apply'] && typeof e['apply'] === 'object') {
      const apply = e['apply'] as Record<string, unknown>;
      if (typeof apply['function'] === 'string' && fnNames.has(apply['function'])) {
        const calls = callGraph.get(callee) ?? new Set<string>();
        calls.add(apply['function'] as string);
        callGraph.set(callee, calls);
      }
      if (Array.isArray(apply['arguments'])) {
        for (const arg of apply['arguments']) collectCalls(arg, callee);
      }
    }
  }

  for (const fn of fns) {
    if (!fn.name) continue;
    callGraph.set(fn.name, new Set());
    collectCalls(fn.body, fn.name);
  }

  // DFS cycle detection
  const visiting = new Set<string>();
  const visited = new Set<string>();
  function dfs(name: string): boolean {
    if (visiting.has(name)) return true;
    if (visited.has(name)) return false;
    visiting.add(name);
    for (const callee of callGraph.get(name) ?? []) {
      if (dfs(callee)) return true;
    }
    visiting.delete(name);
    visited.add(name);
    return false;
  }
  for (const fn of fns) {
    if (fn.name && dfs(fn.name)) {
      err(issues, `functions[${fn.name}]`, `Recursive function call detected in: ${fn.name}`);
    }
  }
}

function checkOutputResolution(model: OMLEModel, issues: ValidationIssue[]) {
  const available = new Set((model.inputs ?? []).map(i => i.name));
  const schema = model.model_schema;
  if (schema?.features) {
    for (const f of expandFeatures(schema.features)) {
      if (f.name) available.add(f.name);
    }
  }
  for (const node of model.nodes ?? []) {
    for (const out of node.outputs ?? []) {
      if (out.name) available.add(out.name);
    }
  }
  for (const [i, out] of (model.outputs ?? []).entries()) {
    if (out.name && !available.has(out.name)) {
      err(issues, `outputs[${i}].name`,
        `Output "${out.name}" does not resolve to any InputSpec or NodeOutput`);
    }
  }
}
