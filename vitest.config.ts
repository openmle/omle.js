import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.ts'],
    tsconfig: './tsconfig.test.json',
  },
});
