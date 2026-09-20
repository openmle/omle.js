# omle.js

TypeScript library for loading, parsing, validating, and executing [OMLE](https://github.com/openmle) models. Works in Node.js and the browser. One runtime dependency (`protobufjs`, for binary `.omle` decoding).

## Features

- **Full IR** — TypeScript types for every message in `omle.proto`, following the Python SDK's JSON conventions (enums as strings, flat typed-data arrays, base64 bytes).
- **I/O** — `fromJSON` / `toJSON`, `fromProtoBinary` for binary `.omle` files, Node.js `loadFile` / `saveFile`, browser `loadBlob`.
- **Resolution** — builds a tensor-entry index, topologically sorts the DAG, and expands `NameRange` patterns.
- **Validation** — structural and semantic checks with per-path error and warning messages.
- **Reference engine** — executes models in topological order with full composite-node scoping. Supports all structured bodies: decision tree, tree ensemble, linear, neural network (dense layers), naïve Bayes (Gaussian / Multinomial / Bernoulli / Categorical), prototype clustering, GMM, and linear and kernel SVM.
- **Step-by-step inference** — `runWithSteps()` captures input/output snapshots at each node for debugging and visualization.

## Installation

```bash
npm install @openmle/omle.js
```

> **Node ≥ 18, ESM only.** The package ships ES2022 modules with full `.d.ts` declarations.

## Quick start

```ts
import { fromJSON, validate, Engine } from '@openmle/omle.js';
import { readFileSync } from 'node:fs';

const model = fromJSON(readFileSync('model.json', 'utf8'));

const result = validate(model);
if (!result.valid) {
  for (const err of result.errors) console.error(`[${err.path}] ${err.message}`);
}

const engine = new Engine(model);

// Single forward pass — inputs accept scalars, 1-D arrays, 2-D arrays, or TensorData
const outputs = engine.run({ x0: 1.5, x1: 0.3, x2: -0.7 });
console.log(outputs);
// { score: { dtype: 'FLOAT64', shape: [1], data: Float64Array [0.812] } }
```

## API

### I/O

```ts
fromJSON(json: string | object): OMLEModel
toJSON(model: OMLEModel, pretty?: boolean): string

// Binary .omle files (protobuf wire format). The schema is embedded, so
// nothing is fetched or compiled at build time.
fromProtoBinary(data: ArrayBuffer | Uint8Array): OMLEModel

// Node.js only
loadFile(path: string): Promise<OMLEModel>
saveFile(model: OMLEModel, path: string, pretty?: boolean): Promise<void>

// Browser
loadBlob(blob: { text(): Promise<string> }): Promise<OMLEModel>
```

### Validation

```ts
validate(model: OMLEModel): ValidationResult

interface ValidationResult {
  valid: boolean;
  errors:   Array<{ path: string; message: string }>;
  warnings: Array<{ path: string; message: string }>;
}
```

### Resolution

```ts
resolve(model: OMLEModel): ResolvedModel

interface ResolvedModel {
  model: OMLEModel;
  tensorIndex: Map<string, TensorEntry>;   // lookup by TensorRef.id
  executionOrder: Node[];                  // topologically sorted
  schemaNames: Set<string>;                // inputs + schema features
}

expandNameRange(range: NameRange): string[]
expandNodeInputs(inputs: NodeInput[]): string[]
expandFeatures(features: Feature[]): Feature[]
```

### Engine

```ts
const engine = new Engine(model);

// Standard forward pass
engine.run(inputs: InferenceInput): InferenceOutput

// Step-by-step — returns per-node snapshots for debugging / visualization
engine.runWithSteps(inputs: InferenceInput): SteppedResult
```

**`InferenceInput`** accepts any mix of:
| Value | Interpreted as |
|---|---|
| `1.5` | scalar `FLOAT64[1]` |
| `[1, 2, 3]` | 1-D `FLOAT64[3]` |
| `[[1, 2], [3, 4]]` | 2-D `FLOAT64[2×2]` |
| `TensorData` | passed through (converts plain `number[]` `.data` to `Float64Array`) |
| `Tensor` (IR) | converted via `tensorToData()` |

```ts
interface StepSnapshot {
  nodeName: string;
  nodeId:   string;          // "node:<name>"
  inputs:   Record<string, SerializedTensor>;
  outputs:  Record<string, SerializedTensor>;
  warnings: string[];
}

interface SerializedTensor {
  dtype: string;
  shape: number[];
  data:  (number | string | boolean)[] | null;  // null = value not produced
}

interface SteppedResult {
  output: InferenceOutput;
  steps:  StepSnapshot[];
}
```

### Ops utilities

```ts
import { tensorToData, applyPostTransform } from '@openmle/omle.js';

// Convert an IR Tensor to the engine's internal TensorData
tensorToData(tensor: Tensor): TensorData

// Apply a post-transform in-place over a Float64Array
applyPostTransform(
  scores: Float64Array,
  transform: PostTransform | undefined,
  outputSize: number,
): Float64Array
```

## JSON conventions

This library follows the Python reference SDK's serialization rules:

- Enums are their **string names** (`"FLOAT32"`, not `11`).
- Absent and empty fields are **omitted** on write.
- Bytes fields are **base64-encoded** strings.
- Typed tensor payloads are **flat arrays** (`float32_data: [1,2,3]`, not `{values:[1,2,3]}`).
- `int64` values are JavaScript `number` (safe up to 2⁵³).

## Development

```bash
npm run build      # compile to dist/
npm run dev        # watch mode
npm run typecheck  # type-check without emit
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the checks a
change needs to pass, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community
expectations.

## License

Apache-2.0
