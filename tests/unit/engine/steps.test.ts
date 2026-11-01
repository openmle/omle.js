// Unit tests for Engine.runWithSteps() — step capture, including the interior
// of composite nodes.

import { describe, test, expect } from 'vitest';
import { Engine } from '../../../src/engine/executor.js';
import type { OMLEModel, Node } from '../../../src/ir.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Elementwise scale node: out = in * factor.
 *
 * WeightedSum resolves `weights` through getTensorData(), so it has to be a
 * tensor attribute — an inline `floats` list is ignored and the node produces
 * no output at all.
 */
function scaleNode(name: string, inName: string, outName: string, factor: number): Node {
  return {
    name,
    domain: 'omle.core',
    op: 'WeightedSum',
    inputs: [{ name: { value: inName } }],
    outputs: [{ name: outName }],
    attributes: [{
      name: 'weights',
      tensor: { type: { dtype: 'FLOAT64', shape: [1] }, float64_data: [factor] },
    }],
  } as Node;
}

function run(model: OMLEModel, inputs: Record<string, unknown>) {
  return new Engine(model).runWithSteps(inputs as never);
}

// ── Flat models ───────────────────────────────────────────────────────────────

describe('runWithSteps — flat model', () => {
  test('one step per node, no children', () => {
    const model: OMLEModel = {
      inputs: [{ name: 'x' }],
      outputs: [{ name: 'b' }],
      nodes: [scaleNode('n1', 'x', 'a', 2), scaleNode('n2', 'a', 'b', 3)],
    };
    const { steps, output } = run(model, { x: { dtype: 'FLOAT64', shape: [1, 1], data: [5] } });

    expect(steps.map(s => s.nodeName)).toEqual(['n1', 'n2']);
    expect(steps.every(s => s.children === undefined)).toBe(true);
    expect(Array.from(output['b'].data as Float64Array)).toEqual([30]);
  });

  test('each step carries its node id, inputs and outputs', () => {
    const model: OMLEModel = {
      inputs: [{ name: 'x' }],
      outputs: [{ name: 'a' }],
      nodes: [scaleNode('n1', 'x', 'a', 2)],
    };
    const { steps } = run(model, { x: { dtype: 'FLOAT64', shape: [1, 1], data: [4] } });

    expect(steps[0].nodeId).toBe('node:n1');
    expect(Object.keys(steps[0].inputs)).toEqual(['x']);
    expect(steps[0].outputs['a'].data).toEqual([8]);
    expect(steps[0].warnings).toEqual([]);
  });
});

// ── Composite models ──────────────────────────────────────────────────────────

describe('runWithSteps — composite interior', () => {
  /** Composite wrapping two scale nodes: inner = x*2, then out = inner*3. */
  function compositeModel(): OMLEModel {
    return {
      inputs: [{ name: 'x' }],
      outputs: [{ name: 'c_out' }],
      nodes: [
        {
          name: 'comp',
          domain: 'omle.core',
          op: 'Composite',
          inputs: [{ name: { value: 'x' } }],
          outputs: [{ name: 'c_out' }],
          composite: {
            nodes: [scaleNode('inner1', 'x', 'mid', 2), scaleNode('inner2', 'mid', 'c_out', 3)],
          },
        } as Node,
      ],
    };
  }

  test('the top-level array still has one entry per top-level node', () => {
    // Interior steps hang off the composite rather than being spliced into the
    // top level, so a caller that only replays the outer graph is unaffected.
    const { steps } = run(compositeModel(), { x: { dtype: 'FLOAT64', shape: [1, 1], data: [5] } });
    expect(steps).toHaveLength(1);
    expect(steps[0].nodeName).toBe('comp');
  });

  test('the composite step carries its interior steps in execution order', () => {
    const { steps } = run(compositeModel(), { x: { dtype: 'FLOAT64', shape: [1, 1], data: [5] } });
    const children = steps[0].children;
    expect(children).toBeDefined();
    expect(children!.map(s => s.nodeName)).toEqual(['inner1', 'inner2']);
    // x*2 = 10, then *3 = 30
    expect(children![0].outputs['mid'].data).toEqual([10]);
    expect(children![1].outputs['c_out'].data).toEqual([30]);
  });

  test('interior ids use the plain node: scheme, scoped to the composite', () => {
    // The viewer builds a composite's interior as its own synthetic model, so
    // inner nodes are named node:<name> there too — matching these ids means
    // no translation is needed to line steps up with the drilled-in graph.
    const { steps } = run(compositeModel(), { x: { dtype: 'FLOAT64', shape: [1, 1], data: [1] } });
    expect(steps[0].children!.map(s => s.nodeId)).toEqual(['node:inner1', 'node:inner2']);
  });

  test('nested composites recurse to any depth', () => {
    const model: OMLEModel = {
      inputs: [{ name: 'x' }],
      outputs: [{ name: 'outer_out' }],
      nodes: [
        {
          name: 'outer',
          domain: 'omle.core',
          op: 'Composite',
          inputs: [{ name: { value: 'x' } }],
          outputs: [{ name: 'outer_out' }],
          composite: {
            nodes: [
              {
                name: 'inner_comp',
                domain: 'omle.core',
                op: 'Composite',
                inputs: [{ name: { value: 'x' } }],
                outputs: [{ name: 'outer_out' }],
                composite: { nodes: [scaleNode('deep', 'x', 'outer_out', 4)] },
              } as Node,
            ],
          },
        } as Node,
      ],
    };
    const { steps } = run(model, { x: { dtype: 'FLOAT64', shape: [1, 1], data: [2] } });
    expect(steps[0].nodeName).toBe('outer');
    expect(steps[0].children!.map(s => s.nodeName)).toEqual(['inner_comp']);
    expect(steps[0].children![0].children!.map(s => s.nodeName)).toEqual(['deep']);
    expect(steps[0].children![0].children![0].outputs['outer_out'].data).toEqual([8]);
  });

  test('a composite with no interior nodes reports no children', () => {
    const model: OMLEModel = {
      inputs: [{ name: 'x' }],
      outputs: [{ name: 'x' }],
      nodes: [{
        name: 'empty', domain: 'omle.core', op: 'Composite',
        inputs: [{ name: { value: 'x' } }], outputs: [{ name: 'x' }],
        composite: { nodes: [] },
      } as Node],
    };
    const { steps } = run(model, { x: { dtype: 'FLOAT64', shape: [1, 1], data: [1] } });
    expect(steps[0].children).toBeUndefined();
  });
});
