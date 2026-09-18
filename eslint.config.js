// ESLint flat config.
//
// Scope mirrors the Python side: correctness rules enforced now, stylistic
// rules left to review. No formatter — `tsc` and these rules do not rewrite
// source, so existing formatting is preserved.

import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: ['dist/**', 'node_modules/**', 'coverage/**', '**/*.d.ts'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      parserOptions: { ecmaVersion: 2022, sourceType: 'module' },
      globals: { console: 'readonly', process: 'readonly', Buffer: 'readonly' },
    },
    rules: {
      // Prefix-underscore is the agreed opt-out for intentionally unused names.
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrorsIgnorePattern: '^_' },
      ],
      // TypeScript resolves identifiers itself; the core rule only produces
      // false positives on typed sources.
      'no-undef': 'off',
      // Defensive `let x = null` before a branch chain is idiomatic here and
      // keeps TypeScript's definite-assignment analysis happy.
      'no-useless-assignment': 'off',
    },
  },
  {
    // Ad-hoc debugging tests poke at internals; `any` is the point there.
    files: ['tests/debug_*.test.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-non-null-asserted-optional-chain': 'off',
    },
  },
);
