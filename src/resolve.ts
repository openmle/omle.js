// Reference resolution for OMLE models.
//
// Builds lookup structures and expands NameRange patterns so the engine and
// inspector never have to deal with indirection or inline expansion logic.

import type {
  OMLEModel, TensorEntry, TensorRef, Tensor, SparseTensor, TensorValue,
  NameRange, Feature, NodeInput, Node,
} from './ir.js';

// ── Public types ──────────────────────────────────────────────────────────────

export interface ResolvedModel {
  model: OMLEModel;
  /** TensorEntry lookup by id */
  tensorIndex: Map<string, TensorEntry>;
  /** Topological execution order for top-level nodes */
  executionOrder: Node[];
  /** Names visible after ModelSchema preprocessing (inputs + schema features) */
  schemaNames: Set<string>;
}

// ── Main entry point ──────────────────────────────────────────────────────────

export function resolve(model: OMLEModel): ResolvedModel {
  const tensorIndex = buildTensorIndex(model);
  const schemaNames = buildSchemaNames(model);
  const executionOrder = topoSort(model.nodes ?? [], schemaNames);
  return { model, tensorIndex, executionOrder, schemaNames };
}

// ── TensorRef resolution ──────────────────────────────────────────────────────

export function resolveRef(ref: TensorRef, index: Map<string, TensorEntry>): TensorEntry {
  const entry = index.get(ref.id);
  if (!entry) throw new Error(`TensorRef not found: ${ref.id}`);
  return entry;
}

export function resolveRefDense(ref: TensorRef, index: Map<string, TensorEntry>): Tensor {
  const entry = resolveRef(ref, index);
  if (!entry.dense) throw new Error(`TensorEntry "${ref.id}" is not a dense tensor`);
  return entry.dense;
}

export function resolveRefSparse(ref: TensorRef, index: Map<string, TensorEntry>): SparseTensor {
  const entry = resolveRef(ref, index);
  if (!entry.sparse) throw new Error(`TensorEntry "${ref.id}" is not a sparse tensor`);
  return entry.sparse;
}

export function resolveTensorValue(
  tv: TensorValue | null | undefined,
  index: Map<string, TensorEntry>,
): Tensor | null {
  if (!tv) return null;
  if (tv.tensor) return tv.tensor;
  if (tv.tensor_ref) return resolveRefDense(tv.tensor_ref, index);
  return null;
}

export function resolveTensorValueSparse(
  tv: TensorValue | null | undefined,
  index: Map<string, TensorEntry>,
): SparseTensor | null {
  if (!tv) return null;
  if (tv.sparse) return tv.sparse;
  if (tv.tensor_ref) {
    const entry = index.get(tv.tensor_ref.id);
    if (entry?.sparse) return entry.sparse;
  }
  return null;
}

// ── NameRange expansion ───────────────────────────────────────────────────────

export function expandNameRange(range: NameRange): string[] {
  const names: string[] = [];
  const width = range.width ?? 0;
  for (let i = range.start; i < range.end; i++) {
    const suffix = width > 0 ? String(i).padStart(width, '0') : String(i);
    names.push(`${range.prefix}${suffix}`);
  }
  return names;
}

export function expandNodeInputs(inputs: NodeInput[]): string[] {
  const names: string[] = [];
  for (const inp of inputs) {
    if (inp.name !== undefined) {
      names.push(inp.name.value);
    } else if (inp.range !== undefined) {
      names.push(...expandNameRange(inp.range));
    }
  }
  return names;
}

export function expandFeatures(features: Feature[]): Feature[] {
  const expanded: Feature[] = [];
  for (const f of features) {
    if (f.name !== undefined) {
      expanded.push(f);
    } else if (f.range !== undefined) {
      const names = expandNameRange(f.range);
      names.forEach((name, i) => {
        expanded.push({
          ...f,
          range: undefined,
          name,
          index: f.index !== undefined ? f.index + i : i,
        });
      });
    }
  }
  return expanded;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function buildTensorIndex(model: OMLEModel): Map<string, TensorEntry> {
  const idx = new Map<string, TensorEntry>();
  for (const entry of model.tensor_entries ?? []) {
    idx.set(entry.id, entry);
  }
  return idx;
}

function buildSchemaNames(model: OMLEModel): Set<string> {
  const names = new Set<string>();
  for (const inp of model.inputs ?? []) names.add(inp.name);
  const schema = model.model_schema;
  if (schema?.features) {
    for (const f of expandFeatures(schema.features)) {
      if (f.name) names.add(f.name);
    }
  }
  return names;
}

function topoSort(nodes: Node[], initialNames: Set<string>): Node[] {
  const available = new Set(initialNames);
  const remaining = [...nodes];
  const ordered: Node[] = [];
  let changed = true;

  while (changed && remaining.length > 0) {
    changed = false;
    for (let i = remaining.length - 1; i >= 0; i--) {
      const node = remaining[i];
      const inputNames = expandNodeInputs(node.inputs ?? []);
      if (inputNames.every(n => available.has(n))) {
        ordered.push(node);
        for (const out of node.outputs ?? []) available.add(out.name);
        remaining.splice(i, 1);
        changed = true;
      }
    }
  }

  // Any remaining nodes have unresolved inputs — append in original order
  ordered.push(...remaining);
  return ordered;
}
