# Contributing to omle.js

Thanks for your interest. This document covers how to work on this repository —
setup, the layout, and the checks a change needs to pass.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## What lives here

A TypeScript library that loads, validates and **executes** OMLE models in the
browser or Node — the second implementation of the execution semantics, next to
the C++ [omle-runtime](https://github.com/openmle/omle-runtime).

The format is defined in [omle](https://github.com/openmle/omle): the proto
schema, the operator registry, and what each operator means. A change to an
operator's *semantics* belongs there; this repository implements it, and should
agree numerically with omle-runtime.

## Getting started

```bash
npm ci

npm run typecheck    # tsc --noEmit
npm run lint         # eslint .
npm test             # vitest run
npm run build        # tsc → dist/
```

`npm test` runs all unit tests in well under a second.

## Layout

```
src/
  index.ts        public entry point
  io.ts           model loading and decoding
  ir.ts           TypeScript types mirroring the proto schema
  validate.ts     structural validation
  resolve.ts      tensor/name resolution
  engine/
    executor.ts   graph execution — the DAG walk and node dispatch
    tree.ts       tree and tree-ensemble scoring
    linear.ts, svm.ts, nn.ts, naive_bayes.ts, clustering.ts
    preprocess.ts omle.feature operators
    ops.ts        omle.core operators
    predicates.ts predicate evaluation
    explain.ts    per-feature contributions
tests/unit/
  ops.test.ts, io.test.ts, resolve.test.ts
  engine/
    predicates.test.ts, preprocess.test.ts, models.test.ts
    tree.test.ts, nn.test.ts, svm.test.ts
    clustering.test.ts, validate_inputs.test.ts
tools/
  verify_engine.ts  CLI tool for end-to-end verification against model files
                    (run with: npm run verify -- --models-dir <dir>)
scripts/
  gen_ir_ts.py    generates src/ir.ts from omle.proto
```

## The proto package name

Models are decoded with protobufjs against `omle.proto`, whose package is
**`omle`**:

```ts
const ModelType = root.lookupType('omle.OMLEModel');
```

There is no `omle.v1`. Looking up a wrong type name throws
`no such type: …` at module load, which vitest reports as
`No test suite found in file …` — the tests in that file then never register and
the run can look deceptively green. If a test file's count drops unexpectedly,
check for that error first.

## Adding an operator

1. Implement it in the right `src/engine/` module and wire it into the dispatch
   in `executor.ts`.
2. Match the contract in the `omle` registry exactly — input kinds, attribute
   names, and output shape/dtype rules. Handle optional attributes being absent;
   `StandardScaler` skipping centering when `mean` is missing is the pattern.
3. Add unit tests under `tests/unit/engine/`.
4. The same operator in omle-runtime should agree numerically. Note in your
   PR if it needs a matching change.

## Code style

ESLint (`eslint.config.js`) and `tsc` both run in CI. Neither rewrites source,
so existing formatting is preserved — there is no Prettier step.

- Prefix intentionally unused names with `_`; that is the configured opt-out.
- `no-undef` is off: TypeScript resolves identifiers itself and the core rule
  only produces false positives on typed sources.
- `any` is not permitted in test files; keep types explicit.

## Pre-commit hooks (optional)

```bash
pip install pre-commit
pre-commit install
```

Runs ESLint and `tsc --noEmit` through this repository's own toolchain, so
`npm ci` must have run first.

## Tests and CI

`.github/workflows/test.yml` runs lint, typecheck and the suite on every push
and pull request.

Please add a test with any behaviour change. For a bug fix, a test that fails
before the fix is the most useful thing you can include. When a fix corrects a
numeric result, state the expected value's source in the test — the framework
that trained the model, or omle-runtime.

## Reporting bugs

Include the model file (or the script that produced it), the input that triggers
it, and the expected versus actual output. If omle-runtime scores the same
model differently, say so — a disagreement between the two engines is the most
important kind of bug this project can have.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE), in
accordance with section 5 of that license. There is no separate CLA.
