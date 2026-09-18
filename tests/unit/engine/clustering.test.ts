// Unit tests for engine/clustering.ts — prototype (k-means) and GMM clustering.

import { describe, test, expect } from 'vitest';
import { executeClustering } from '../../../src/engine/clustering.js';
import type { Clustering } from '../../../src/ir.js';
import type { ResolvedModel } from '../../../src/resolve.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function emptyResolved(): ResolvedModel {
  return { model: { inputs: [] }, tensorIndex: new Map(), executionOrder: [], schemaNames: new Set() };
}

function f64(...values: number[]): Float64Array {
  return new Float64Array(values);
}

function tv(data: number[], shape: number[]) {
  return { tensor: { type: { dtype: 'FLOAT64', shape }, float64_data: data } };
}

// ── PrototypeClustering (k-means) ─────────────────────────────────────────────

describe('executeClustering — PrototypeClustering EUCLIDEAN', () => {
  // 2 clusters: center0=[0,0], center1=[10,10]
  function kmeans(): Clustering {
    return {
      prototype: {
        centers: tv([0, 0,  10, 10], [2, 2]),
        distance_measure: 'EUCLIDEAN',
      },
    };
  }

  test('point near center 0 assigned to cluster 0', () => {
    const { labels, distances } = executeClustering(kmeans(), f64(1, 1), 2, 1, emptyResolved());
    expect(labels[0]).toBe(0);
    expect(distances[0]).toBeCloseTo(Math.sqrt(2), 6);
  });

  test('point near center 1 assigned to cluster 1', () => {
    const { labels, distances } = executeClustering(kmeans(), f64(9, 9), 2, 1, emptyResolved());
    expect(labels[0]).toBe(1);
    expect(distances[0]).toBeCloseTo(Math.sqrt(2), 6);
  });

  test('batch: two points routed to different clusters', () => {
    const { labels } = executeClustering(kmeans(), f64(1, 1,  9, 9), 2, 2, emptyResolved());
    expect(labels[0]).toBe(0);
    expect(labels[1]).toBe(1);
  });

  test('point exactly at center: distance = 0', () => {
    const { labels, distances } = executeClustering(kmeans(), f64(0, 0), 2, 1, emptyResolved());
    expect(labels[0]).toBe(0);
    expect(distances[0]).toBeCloseTo(0, 10);
  });
});

describe('executeClustering — PrototypeClustering distance measures', () => {
  // center0=[0,0], center1=[3,4]
  function clusteringWith(measure: string): Clustering {
    return {
      prototype: {
        centers: tv([0, 0,  3, 4], [2, 2]),
        distance_measure: measure,
      },
    };
  }

  test('MANHATTAN: |Δx|+|Δy|', () => {
    // x=[1,1]: d0=2, d1=|3-1|+|4-1|=5 → cluster 0
    const { labels, distances } = executeClustering(clusteringWith('MANHATTAN'), f64(1, 1), 2, 1, emptyResolved());
    expect(labels[0]).toBe(0);
    expect(distances[0]).toBeCloseTo(2, 10);
  });

  test('SQUARED_EUCLIDEAN: squared L2 distance', () => {
    // x=[0,0]: d0=0, d1=3^2+4^2=25 → cluster 0, dist=0
    const { distances } = executeClustering(clusteringWith('SQUARED_EUCLIDEAN'), f64(0, 0), 2, 1, emptyResolved());
    expect(distances[0]).toBeCloseTo(0, 10);
  });

  test('COSINE: 1 - cos similarity', () => {
    // x=[3,4] exactly matches center1 → cosine similarity = 1 → distance = 0
    const { labels, distances } = executeClustering(clusteringWith('COSINE'), f64(3, 4), 2, 1, emptyResolved());
    expect(labels[0]).toBe(1);
    expect(distances[0]).toBeCloseTo(0, 6);
  });
});

// ── GaussianMixtureClustering (diagonal covariance) ──────────────────────────

describe('executeClustering — GaussianMixtureClustering (diagonal)', () => {
  // 2 components, 1 feature:
  //   component 0: weight=0.5, mean=[0], var=[1]
  //   component 1: weight=0.5, mean=[10], var=[1]
  function gmm(): Clustering {
    return {
      gaussian_mixture: {
        weights:     tv([0.5, 0.5], [2]),
        means:       tv([0, 10], [2, 1]),
        covariances: tv([1, 1], [2, 1]),
        covariance_type: 'DIAGONAL',
      },
    };
  }

  test('point near mean 0 assigned to component 0', () => {
    const { labels } = executeClustering(gmm(), f64(1), 1, 1, emptyResolved());
    expect(labels[0]).toBe(0);
  });

  test('point near mean 10 assigned to component 1', () => {
    const { labels } = executeClustering(gmm(), f64(9), 1, 1, emptyResolved());
    expect(labels[0]).toBe(1);
  });

  test('batch: two points routed independently', () => {
    const { labels } = executeClustering(gmm(), f64(1, 9), 1, 2, emptyResolved());
    expect(labels[0]).toBe(0);
    expect(labels[1]).toBe(1);
  });
});

// ── Fallthrough / empty ───────────────────────────────────────────────────────

describe('executeClustering — fallthrough', () => {
  test('clustering with no variant returns zero labels and distances', () => {
    const { labels, distances } = executeClustering({}, f64(1, 2), 2, 1, emptyResolved());
    expect(labels[0]).toBe(0);
    expect(distances[0]).toBe(0);
  });
});
