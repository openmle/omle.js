// Unit tests for src/validate.ts — metadata.format_version handling.
//
// The OMLE format is pre-1.0: omle.proto documents "0.1.0" as its example and
// every model the SDK writes carries a 0.x version, so rejecting major 0 would
// reject the entire ecosystem.

import { describe, test, expect } from 'vitest';
import { validate } from '../../src/validate.js';
import type { OMLEModel } from '../../src/ir.js';

function issuesFor(format_version?: string) {
  const model = { metadata: { name: 'm', format_version } } as OMLEModel;
  const result = validate(model);
  const at = (s: 'error' | 'warning') =>
    (s === 'error' ? result.errors : result.warnings)
      .filter(i => i.path === 'metadata.format_version');
  return { result, errors: at('error'), warnings: at('warning') };
}

describe('format_version', () => {
  test('0.x is valid — the version every current model carries', () => {
    const { result, errors } = issuesFor('0.1.0');
    expect(errors).toEqual([]);
    expect(result.valid).toBe(true);
  });

  test('1.x is valid, ahead of the v1 release', () => {
    expect(issuesFor('1.0.0').errors).toEqual([]);
  });

  test('an unrecognized major warns rather than failing the model', () => {
    const { result, errors, warnings } = issuesFor('2.0.0');
    expect(errors).toEqual([]);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]?.message).toContain('2');
    // Still loadable: the Python SDK accepts any non-empty version.
    expect(result.valid).toBe(true);
  });

  test('a malformed version is an error', () => {
    const { errors } = issuesFor('not-a-version');
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toContain('Malformed');
  });

  test('a missing version is an error, matching the Python SDK', () => {
    const { result, errors } = issuesFor(undefined);
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toContain('must not be empty');
    expect(result.valid).toBe(false);
  });
});
