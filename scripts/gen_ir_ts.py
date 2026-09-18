#!/usr/bin/env python3
"""Generate src/ir.ts from omle.proto in the omle GitHub repo.

Fetches the proto from https://github.com/openmle/omle at a configurable
ref (branch, tag, or commit SHA), compiles it on-the-fly with grpcio-tools,
then uses protobuf reflection to emit TypeScript types.

Requirements:
  pip install grpcio-tools

Usage (from the omle.js repo root):
  python3 scripts/gen_ir_ts.py             # writes src/ir.ts  (ref: main)
  python3 scripts/gen_ir_ts.py <out>       # writes to <out>
  OMLE_PROTO_REF=v1.2.0 npm run gen:ir  # pin to a release tag

TypeScript output conventions (match Python SDK JSON serialisation):
  - enums  → string-literal union types  ('FLOAT32' | 'FLOAT64' | ...)
  - oneof  → individual optional fields  (not discriminated unions)
  - maps   → Record<K, V>
  - bytes  → string  (base64)
  - int64  → number  (safe up to 2^53)
  - most fields optional; a curated set of "logically required" fields kept non-optional
"""

import sys
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# ── Locate omle.proto ──────────────────────────────────────────────────────
# Resolution order:
#   1. OMLE_PROTO_PATH env var — explicit local path (dev / pre-push)
#   2. git clone from GitHub       — default for CI and released state
#      Auth: SSH key (local dev) or GITHUB_TOKEN env var (CI)
#      Ref:  OMLE_PROTO_REF env var, default "main"

PROTO_REPO = "openmle/omle"
PROTO_PATH = "protobuf/omle.proto"
PROTO_REF  = os.environ.get("OMLE_PROTO_REF", "main")

local_override = os.environ.get("OMLE_PROTO_PATH")

if local_override:
    _proto_file = Path(local_override).expanduser().resolve()
    if not _proto_file.exists():
        print(f"Error: OMLE_PROTO_PATH={local_override!r} does not exist.", file=sys.stderr)
        sys.exit(1)
    print(f"Using local proto: {_proto_file}", file=sys.stderr)
else:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        remote = f"https://x-access-token:{token}@github.com/{PROTO_REPO}.git"
        remote_display = f"https://github.com/{PROTO_REPO}.git  (token auth)"
    else:
        remote = f"git@github.com:{PROTO_REPO}.git"
        remote_display = remote

    print(f"Cloning {PROTO_REPO}@{PROTO_REF} to get {PROTO_PATH}", file=sys.stderr)
    _clone_dir = tempfile.mkdtemp(prefix="omle_proto_")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", PROTO_REF, remote, _clone_dir],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: git clone failed:\n{e.stderr.decode()}", file=sys.stderr)
        print(f"  Remote: {remote_display}", file=sys.stderr)
        print(f"  Ref:    {PROTO_REF}", file=sys.stderr)
        print("  Tip: set OMLE_PROTO_PATH=/path/to/omle.proto to use a local copy.", file=sys.stderr)
        sys.exit(1)

    _proto_file = Path(_clone_dir) / PROTO_PATH
    if not _proto_file.exists():
        print(f"Error: {PROTO_PATH} not found in cloned repo.", file=sys.stderr)
        print("  Tip: set OMLE_PROTO_PATH=/path/to/omle.proto to use a local copy.", file=sys.stderr)
        sys.exit(1)

# ── Compile proto → omle_pb2.py in a temp dir ──────────────────────────────
_tmp = tempfile.mkdtemp(prefix="omle_pb2_")
subprocess.run(
    [sys.executable, "-m", "grpc_tools.protoc",
     f"-I{_proto_file.parent}",
     f"--python_out={_tmp}",
     str(_proto_file)],
    check=True,
    capture_output=True,
)
sys.path.insert(0, _tmp)
import omle_pb2 as _pb2  # noqa: E402  (generated into _tmp)

from google.protobuf.descriptor import (
    FieldDescriptor, Descriptor, EnumDescriptor, FileDescriptor,
)

FILE_DESC: FileDescriptor = _pb2.DESCRIPTOR

# ── List-wrapper detection ─────────────────────────────────────────────────────
# Messages with exactly one field named "values" that is REPEATED are transparent
# array wrappers (flattenListWrappers in io.ts unwraps them at runtime).

def _list_wrapper_element_ts(msg: Descriptor) -> Optional[str]:
    fields = list(msg.fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        is_rep = (len(fields) == 1 and fields[0].name == "values"
                  and fields[0].label == FieldDescriptor.LABEL_REPEATED)
    if not is_rep:
        return None
    f = fields[0]
    if f.message_type:
        if f.message_type.GetOptions().map_entry:
            return None
        inner = _list_wrapper_element_ts(f.message_type)
        return inner  # None if nested message that isn't itself a wrapper
    return _SCALAR_MAP.get(f.type)

_WRAPPER_CACHE: dict[str, Optional[str]] = {}

def _resolve_wrapper(msg: Descriptor) -> Optional[str]:
    key = msg.full_name
    if key not in _WRAPPER_CACHE:
        elem = _list_wrapper_element_ts(msg)
        _WRAPPER_CACHE[key] = f"{elem}[]" if elem else None
    return _WRAPPER_CACHE[key]

# ── Scalar type map ────────────────────────────────────────────────────────────
_SCALAR_MAP = {
    FieldDescriptor.TYPE_BOOL:     "boolean",
    FieldDescriptor.TYPE_BYTES:    "string",   # base64
    FieldDescriptor.TYPE_STRING:   "string",
    FieldDescriptor.TYPE_FLOAT:    "number",
    FieldDescriptor.TYPE_DOUBLE:   "number",
    FieldDescriptor.TYPE_INT32:    "number",
    FieldDescriptor.TYPE_INT64:    "number",
    FieldDescriptor.TYPE_UINT32:   "number",
    FieldDescriptor.TYPE_UINT64:   "number",
    FieldDescriptor.TYPE_SINT32:   "number",
    FieldDescriptor.TYPE_SINT64:   "number",
    FieldDescriptor.TYPE_FIXED32:  "number",
    FieldDescriptor.TYPE_FIXED64:  "number",
    FieldDescriptor.TYPE_SFIXED32: "number",
    FieldDescriptor.TYPE_SFIXED64: "number",
}

# ── Rename map ────────────────────────────────────────────────────────────────
# Nested proto types whose short name would clash or differ from the TypeScript
# convention used in ir.ts.  Keys are "ParentMessage.TypeName" (last two
# dot-segments of full_name, package-agnostic).
_RENAME: dict[str, str] = {
    "NeuralNetwork.Activation":          "NNActivation",
    "Tree.NodeKind":                     "TreeNodeKind",
    "Tree.SplitOp":                      "TreeSplitOp",
    "Tree.ComplexPredicate":             "ComplexPredicate",
    "TreeEnsemble.Aggregation":          "TreeEnsembleAggregation",
    "SVM.MulticlassStrategy":            "SVMMulticlassStrategy",
    "CompoundPredicate.BooleanOperator": "CompoundBooleanOp",
    "SimplePredicate.Operator":          "SimplePredicateOp",
    "SimpleSetPredicate.Operator":       "SimpleSetPredicateOp",
    "Interval.Closure":                  "IntervalClosure",
    "DomainValue.ValueProperty":         "DomainValueProperty",
    "DefineFunction.Parameter":          "FunctionParameter",
}

def _ts_name(desc) -> str:
    parts = desc.full_name.split(".")
    if len(parts) >= 2:
        key = f"{parts[-2]}.{parts[-1]}"
        if key in _RENAME:
            return _RENAME[key]
    return desc.name

# ── Required-field override ────────────────────────────────────────────────────
# Proto3 makes all fields optional, but the engine runtime assumes these are
# always present.  Listed as (ts_message_name, field_name).
_REQUIRED: set[tuple[str, str]] = {
    ("Apply",                     "function"),
    ("Attribute",                 "name"),
    ("BernoulliNaiveBayes",       "feature_log_prob"),
    ("CSRMatrix",                 "indices"),
    ("CSRMatrix",                 "indptr"),
    ("CategoricalNaiveBayes",     "category_log_prob"),
    ("ComplexPredicate",          "node_index"),
    ("ComplexPredicate",          "predicate"),
    ("CompoundPredicate",         "op"),
    ("DefineFunction",            "name"),
    ("DenseLayer",                "activation"),
    ("DenseLayer",                "weights"),
    ("FunctionParameter",         "name"),
    ("GaussianMixtureClustering", "covariances"),
    ("GaussianMixtureClustering", "means"),
    ("GaussianMixtureClustering", "weights"),
    ("GaussianNaiveBayes",        "means"),
    ("GaussianNaiveBayes",        "variances"),
    ("InputSpec",                 "name"),
    ("Interval",                  "closure"),
    ("Interval",                  "left_margin"),
    ("Interval",                  "right_margin"),
    ("KernelSVM",                 "dual_coefficients"),
    ("KernelSVM",                 "support_vectors"),
    ("Linear",                    "coefficients"),
    ("LinearSVM",                 "coefficients"),
    ("MultinomialNaiveBayes",     "feature_log_prob"),
    ("NaiveBayes",                "class_log_priors"),
    ("NameAlias",                 "from_name"),
    ("NameAlias",                 "to_name"),
    ("NameRange",                 "end"),
    ("NameRange",                 "prefix"),
    ("NameRange",                 "start"),
    ("NameRef",                   "value"),
    ("NamespaceImport",           "namespace"),
    ("Node",                      "name"),
    ("NodeOutput",                "name"),
    ("OutputSpec",                "name"),
    ("PrototypeClustering",       "centers"),
    ("SimplePredicate",           "column"),
    ("SimplePredicate",           "op"),
    ("SimpleSetPredicate",        "column"),
    ("SimpleSetPredicate",        "op"),
    ("Target",                    "name"),
    ("TensorEntry",               "id"),
    ("TensorRef",                 "id"),
    ("TensorType",                "dtype"),
    ("TensorType",                "shape"),
}

# ── Extra fields injected into specific messages ───────────────────────────────
# Fields that the engine uses but are not yet in omle.proto.
# When the proto is updated these entries should be removed.
_EXTRA_MSG_FIELDS: dict[str, list[str]] = {
    "Node": [
        "  scorecard?: Scorecard;",
        "  ruleset?: RuleSet;",
    ],
}

# ── Field rendering ────────────────────────────────────────────────────────────

def _field_ts_type(field: FieldDescriptor) -> str:
    if field.message_type is not None:
        if field.message_type.GetOptions().map_entry:
            key_ts = _field_ts_type(field.message_type.fields_by_name["key"])
            val_ts = _field_ts_type(field.message_type.fields_by_name["value"])
            return f"Record<{key_ts}, {val_ts}>"
        wrapper = _resolve_wrapper(field.message_type)
        if wrapper:
            return wrapper
        return _ts_name(field.message_type)
    if field.enum_type is not None:
        return _ts_name(field.enum_type)
    return _SCALAR_MAP.get(field.type, "unknown")

def _field_line(field: FieldDescriptor) -> str:
    ts_type = _field_ts_type(field)
    is_map     = field.message_type and field.message_type.GetOptions().map_entry
    is_wrapper = field.message_type and _resolve_wrapper(field.message_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        repeated = field.label == FieldDescriptor.LABEL_REPEATED and not is_map and not is_wrapper
    if repeated:
        ts_type = f"{ts_type}[]"
    parent_ts = _ts_name(field.containing_type)
    sep = "" if (parent_ts, field.name) in _REQUIRED else "?"
    return f"  {field.name}{sep}: {ts_type};"

# ── Collectors ────────────────────────────────────────────────────────────────

def _collect_messages(file_desc: FileDescriptor) -> list[Descriptor]:
    result: list[Descriptor] = []
    queue = list(file_desc.message_types_by_name.values())
    while queue:
        msg = queue.pop(0)
        result.append(msg)
        queue.extend(msg.nested_types)
    return result

def _collect_enums(file_desc: FileDescriptor) -> list[EnumDescriptor]:
    result = list(file_desc.enum_types_by_name.values())
    for msg in _collect_messages(file_desc):
        result.extend(msg.enum_types)
    return result

# ── Emitters ──────────────────────────────────────────────────────────────────

def _emit_enum(e: EnumDescriptor) -> str:
    literals = " | ".join(f"'{v.name}'" for v in e.values)
    return f"export type {_ts_name(e)} =\n  | {literals};\n"

def _emit_message(msg: Descriptor) -> str:
    ts_n  = _ts_name(msg)
    lines = [f"export interface {ts_n} {{"]
    seen_oneofs: set[str] = set()
    for field in msg.fields:
        oo = field.containing_oneof
        if oo and oo.name not in seen_oneofs:
            seen_oneofs.add(oo.name)
            lines.append(f"  // {oo.name} oneof")
        lines.append(_field_line(field))
    for extra in _EXTRA_MSG_FIELDS.get(ts_n, []):
        lines.append(extra)
    lines.append("}")
    return "\n".join(lines) + "\n"

# ── Static blocks ─────────────────────────────────────────────────────────────

PREAMBLE = f"""\
// ──────────────────────────────────────────────────────────────────────────────
// AUTO-GENERATED — do not edit by hand.
// Source: github.com/{PROTO_REPO} @ {PROTO_REF} — {PROTO_PATH}
// Regenerate: npm run gen:ir           (OMLE_PROTO_REF=<tag> to pin a release)
// ──────────────────────────────────────────────────────────────────────────────
//
// OMLE v1 TypeScript IR — mirrors omle.proto field-for-field.
// JSON serialisation convention (from the Python reference SDK):
//   - enums are string names ("FLOAT32", not 11)
//   - absent / empty fields are omitted
//   - bytes fields are base64-encoded strings
//   - typed tensor payload lists (float32_data etc.) are flat arrays, NOT {{values:[]}}
//   - int64 values are represented as number (safe up to 2^53)
"""

# Engine types not yet in omle.proto.  Remove each entry once the proto
# is updated and the field/message is generated automatically above.
STATIC_EXTRAS = """
// ── Types not yet in omle.proto (engine compatibility) ────────────────────

export interface ScorecardAttribute {
  predicate?: Predicate;
  partial_score: Scalar;
}

export interface ScorecardCharacteristic {
  name?: string;
  attributes?: ScorecardAttribute[];
}

export interface Scorecard {
  task_type?: TaskType;
  baseline_score?: Scalar;
  characteristics?: ScorecardCharacteristic[];
  post_transform?: PostTransform;
}

export type RuleSetSelectionMethod =
  | 'SELECTION_METHOD_UNSPECIFIED'
  | 'FIRST_HIT' | 'WEIGHTED_SUM' | 'WEIGHTED_MAX';

export interface Rule {
  id?: string;
  condition?: Predicate;
  score?: Scalar;
  confidence?: Scalar;
  priority?: number;
}

export interface RuleSet {
  task_type?: TaskType;
  rules?: Rule[];
  selection_method?: RuleSetSelectionMethod;
  post_transform?: PostTransform;
}
"""

HELPERS = """
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
"""

# ── Main ──────────────────────────────────────────────────────────────────────

def generate(out_path: Path) -> None:
    all_enums    = _collect_enums(FILE_DESC)
    all_messages = _collect_messages(FILE_DESC)

    chunks = [PREAMBLE]

    # Enums
    chunks.append("// ── Enums " + "─" * 70 + "\n")
    emitted_enums: set[str] = set()
    for e in all_enums:
        ts_n = _ts_name(e)
        if ts_n not in emitted_enums:
            emitted_enums.add(ts_n)
            chunks.append(_emit_enum(e))

    # Messages (skip synthetic MapEntry and transparent list-wrapper types)
    real_msgs = [m for m in all_messages
                 if not m.GetOptions().map_entry and not _resolve_wrapper(m)]
    chunks.append("// ── Messages " + "─" * 67 + "\n")
    emitted_msgs: set[str] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for msg in real_msgs:
            ts_n = _ts_name(msg)
            if ts_n not in emitted_msgs:
                emitted_msgs.add(ts_n)
                chunks.append(_emit_message(msg))

    chunks.append(STATIC_EXTRAS)
    chunks.append(HELPERS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "src" / "ir.ts"
    generate(out)
