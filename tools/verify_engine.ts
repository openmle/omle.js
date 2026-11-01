#!/usr/bin/env node
/**
 * verify_engine — run every verification case embedded in .omle models through
 * the JS engine and report pass/fail.
 *
 * Usage:
 *   npx tsx tools/verify_engine.ts --proto <omle.proto> --models-dir <dir>
 *   npx tsx tools/verify_engine.ts --proto <omle.proto> --model <a.omle> [--model <b.omle>]
 *
 * Options:
 *   --proto      <path>   Path to omle.proto  (required)
 *   --models-dir <path>   Directory of *.omle files to verify
 *   --model      <path>   A single .omle file (repeatable; stacks with --models-dir)
 *   --atol       <n>      Absolute tolerance override  (default: from model or 1e-6)
 *   --rtol       <n>      Relative tolerance override  (default: from model or 1e-5)
 *   --filter     <glob>   Only run models whose stem matches this substring
 *   --help               Show this message
 *
 * Exit code: 0 if all cases pass, 1 if any fail or if the tool itself errors.
 */

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, resolve as resolvePath, basename } from 'node:path';
// protobufjs 8 ships CommonJS, so under ESM `import * as protobuf` yields a
// namespace object whose only keys are `default` and `module.exports` --
// protobuf.parse is then undefined. A default import gets the real module.
import protobuf from 'protobufjs';
import { fromJSON } from '../src/io.js';
import { Engine, tensorToData, scalarToNumber } from '../src/index.js';
import type { OMLEModel, TensorEntry } from '../src/ir.js';
import type { TensorData } from '../src/engine/ops.js';

// ── CLI parsing ───────────────────────────────────────────────────────────────

function parseArgs(argv: string[]): {
  protoPath: string;
  modelFiles: string[];
  atolOverride: number | null;
  rtolOverride: number | null;
  filter: string | null;
} {
  const args = argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log((readFileSync(new URL(import.meta.url).pathname, 'utf-8').match(/\/\*\*([\s\S]*?)\*\//)?.[1] ?? '')
      .replace(/^\s*\* ?/gm, '').trim());
    process.exit(0);
  }

  let protoPath = '';
  const modelFiles: string[] = [];
  let atolOverride: number | null = null;
  let rtolOverride: number | null = null;
  let filter: string | null = null;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    const next = () => { i++; if (i >= args.length) fatal(`${a} requires a value`); return args[i]; };
    if (a === '--proto')      protoPath = resolvePath(next());
    else if (a === '--models-dir') {
      const dir = resolvePath(next());
      if (!existsSync(dir)) fatal(`--models-dir: directory not found: ${dir}`);
      for (const f of readdirSync(dir).filter(f => f.endsWith('.omle')).sort())
        modelFiles.push(join(dir, f));
    }
    else if (a === '--model') modelFiles.push(resolvePath(next()));
    else if (a === '--atol')  atolOverride = Number(next());
    else if (a === '--rtol')  rtolOverride = Number(next());
    else if (a === '--filter') filter = next();
    else fatal(`Unknown option: ${a}`);
  }

  if (!protoPath) fatal('--proto is required');
  if (modelFiles.length === 0) fatal('Provide --models-dir or at least one --model');
  return { protoPath, modelFiles, atolOverride, rtolOverride, filter };
}

function fatal(msg: string): never {
  console.error(`error: ${msg}\nRun with --help for usage.`);
  process.exit(1);
}

// ── Proto / model loading ─────────────────────────────────────────────────────

let _root: protobuf.Root | null = null;

function getRoot(protoPath: string): protobuf.Root {
  if (!_root) {
    const src = readFileSync(protoPath, 'utf-8');
    _root = protobuf.parse(src, { keepCase: true }).root;
  }
  return _root;
}

function flattenListWrappers(val: unknown): unknown {
  if (val === null || val === undefined || typeof val !== 'object') return val;
  if (Array.isArray(val)) return val.map(flattenListWrappers);
  const obj = val as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length === 1 && keys[0] === 'values' && Array.isArray(obj['values'])) {
    const arr = obj['values'] as unknown[];
    if (arr.every(item => item === null || typeof item !== 'object'))
      return arr.map(flattenListWrappers);
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) out[k] = flattenListWrappers(v);
  return out;
}

function loadModel(filePath: string, protoPath: string): OMLEModel {
  const root = getRoot(protoPath);
  const T = root.lookupType('omle.OMLEModel');
  const obj = T.toObject(T.decode(readFileSync(filePath)), {
    longs: Number, enums: String, bytes: String, defaults: false,
  });
  return fromJSON(flattenListWrappers(obj) as object);
}

// ── Comparison ────────────────────────────────────────────────────────────────

function compareTensors(
  actual: TensorData,
  expected: TensorData,
  atol: number,
  rtol: number,
): { pass: boolean; maxAbsErr: number; maxRelErr: number } {
  const a = actual.data;
  const e = expected.data;
  if (a.length !== e.length) return { pass: false, maxAbsErr: Infinity, maxRelErr: Infinity };
  if (expected.dtype === 'STRING') {
    const pass = (a as string[]).every((v, i) => v === (e as string[])[i]);
    return { pass, maxAbsErr: pass ? 0 : Infinity, maxRelErr: pass ? 0 : Infinity };
  }
  let maxAbsErr = 0, maxRelErr = 0, pass = true;
  for (let i = 0; i < a.length; i++) {
    const ai = Number(a[i]), ei = Number(e[i]);
    if (isNaN(ai) || isNaN(ei)) { pass = false; continue; }
    const absErr = Math.abs(ai - ei);
    const relErr = Math.abs(ei) > 0 ? absErr / Math.abs(ei) : absErr;
    if (absErr > maxAbsErr) maxAbsErr = absErr;
    if (relErr > maxRelErr) maxRelErr = relErr;
    if (absErr > atol + rtol * Math.abs(ei)) pass = false;
  }
  return { pass, maxAbsErr, maxRelErr };
}

// ── Runner ────────────────────────────────────────────────────────────────────

interface CaseResult {
  modelStem: string;
  caseLabel: string;
  pass: boolean;
  error?: string;
}

function runModel(
  filePath: string,
  protoPath: string,
  atolOverride: number | null,
  rtolOverride: number | null,
): CaseResult[] {
  const stem = basename(filePath, '.omle');
  let model: OMLEModel;
  try {
    model = loadModel(filePath, protoPath);
  } catch (e) {
    return [{ modelStem: stem, caseLabel: '(load)', pass: false, error: String(e) }];
  }

  const ver = model.verification;
  if (!ver?.cases?.length) return [];  // no verification cases — skip silently

  const index = new Map((model.tensor_entries ?? []).map((te: TensorEntry) => [te.id, te]));
  const results: CaseResult[] = [];

  for (let ci = 0; ci < ver.cases.length; ci++) {
    const vc = ver.cases[ci];
    const caseLabel = ver.cases.length > 1 ? `${stem} [case ${ci}]` : stem;
    const atol = atolOverride ?? (ver.tolerance?.atol != null ? scalarToNumber(ver.tolerance.atol) : 1e-6);
    const rtol = rtolOverride ?? (ver.tolerance?.rtol != null ? scalarToNumber(ver.tolerance.rtol) : 1e-5);

    try {
      const engine = new Engine(model);
      const modelInputs = model.inputs ?? [];
      const inferInputs: Record<string, TensorData> = {};

      for (const [i, ref] of (vc.inputs ?? []).entries()) {
        const entry = index.get(ref.id);
        if (!entry?.dense) throw new Error(`No dense tensor for input ref "${ref.id}"`);
        const name = entry.dense.name ?? modelInputs[i]?.name ?? ref.id;
        inferInputs[name] = tensorToData(entry.dense);
      }

      const output = engine.run(inferInputs);
      let casePassed = true;
      const failures: string[] = [];

      for (const [i, ref] of (vc.expected_outputs ?? []).entries()) {
        const entry = index.get(ref.id);
        if (!entry?.dense) throw new Error(`No dense tensor for expected output ref "${ref.id}"`);
        const outName = entry.dense.name ?? model.outputs?.[i]?.name ?? ref.id;
        const actual = output[outName];
        if (!actual) { casePassed = false; failures.push(`no output "${outName}"`); continue; }
        const { pass, maxAbsErr, maxRelErr } = compareTensors(actual, tensorToData(entry.dense), atol, rtol);
        if (!pass) {
          casePassed = false;
          failures.push(`"${outName}": maxAbsErr=${maxAbsErr.toExponential(3)}, maxRelErr=${maxRelErr.toExponential(3)} (atol=${atol}, rtol=${rtol})`);
        }
      }

      results.push({
        modelStem: stem,
        caseLabel,
        pass: casePassed,
        error: casePassed ? undefined : failures.join('; '),
      });
    } catch (e) {
      results.push({ modelStem: stem, caseLabel, pass: false, error: String(e) });
    }
  }

  return results;
}

// ── Main ──────────────────────────────────────────────────────────────────────

const { protoPath, modelFiles, atolOverride, rtolOverride, filter } = parseArgs(process.argv);

const filtered = filter ? modelFiles.filter(f => basename(f).includes(filter)) : modelFiles;
if (filtered.length === 0) fatal(`No models matched filter "${filter}"`);

let total = 0, passed = 0, failed = 0, skipped = 0;
const failures: CaseResult[] = [];

for (const file of filtered) {
  const results = runModel(file, protoPath, atolOverride, rtolOverride);
  if (results.length === 0) { skipped++; continue; }
  for (const r of results) {
    total++;
    if (r.pass) {
      passed++;
      console.log(`  ✓  ${r.caseLabel}`);
    } else {
      failed++;
      failures.push(r);
      console.log(`  ✗  ${r.caseLabel}`);
      console.log(`       ${r.error}`);
    }
  }
}

console.log('');
console.log(`${total} cases  —  ${passed} passed, ${failed} failed, ${skipped} skipped (no verification cases)`);

if (failed > 0) process.exit(1);
