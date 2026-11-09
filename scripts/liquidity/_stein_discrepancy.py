"""Graph Stein discrepancy: Gorham and Mackey (2015), equations (4), (8).

Use the l1 norm and the standard unit Stein factors. The deterministic graph
starts from nearest-neighbour edges and is augmented until an all-pairs check
certifies a 2-spanner. This has the cited paper's discrepancy equivalence;
we do not claim its construction-time complexity for this implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
import warnings
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist


def verified_spanner(points, neighbours=12, stretch=2.):
    """Return undirected edges with a numerically verified dilation bound."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return np.empty((0, 2), dtype=int), dict(edges=0, maximum_stretch=1.)
    distances = cdist(points, points, metric='cityblock')
    if np.any(distances[np.triu_indices(n, 1)] <= 0):
        raise ValueError('Merge duplicate support points, retaining their weights')
    ranked = np.argsort(distances, axis=1, kind='stable')[:, 1:min(n, neighbours+1)]
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[np.arange(n)[:, None], ranked] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    for iteration in range(3):
        rows, cols = np.nonzero(adjacency)
        graph = csr_matrix((distances[rows, cols], (rows, cols)), shape=(n, n))
        paths = shortest_path(graph, directed=False, method='D')
        violation = paths > stretch*distances + 1e-10
        np.fill_diagonal(violation, False)
        if not violation.any():
            upper = np.triu_indices(n, 1)
            dilation = float(np.max(paths[upper]/distances[upper]))
            edges = np.column_stack(np.nonzero(np.triu(adjacency, 1)))
            return edges, dict(edges=len(edges), maximum_stretch=dilation,
                               neighbours=neighbours, verification_rounds=iteration+1)
        # Adding every violating direct edge yields the required bound on
        # the next pass; all other shortest paths can only decrease.
        adjacency |= violation | violation.T
    raise ValueError('Unable to verify graph dilation')


def graph_constraints(points, edges):
    """Equation (8): bounded values/derivatives and edge Taylor constraints."""
    points, edges = np.asarray(points), np.asarray(edges, dtype=int)
    n, d = points.shape
    u, v = edges[:, 0], edges[:, 1]
    delta = points[u]-points[v]
    distance = abs(delta).sum(axis=1)
    m = len(edges)
    row_chunks, col_chunks, coefficients, bounds = [], [], [], []
    cursor = 0

    def add(columns, values, rhs):
        nonlocal cursor
        columns, values = np.asarray(columns), np.asarray(values)
        count, width = columns.shape
        row_chunks.append(np.repeat(np.arange(cursor, cursor+count), width))
        col_chunks.append(columns.ravel()); coefficients.append(values.ravel())
        bounds.append(np.asarray(rhs)); cursor += count

    for sign in [1., -1.]:
        add(np.column_stack([u, v]), np.tile([sign, -sign], (m, 1)), distance)
        for k in range(d):
            add(np.column_stack([n+u*d+k, n+v*d+k]),
                np.tile([sign, -sign], (m, 1)), distance)
        for endpoint in [u, v]:
            columns = np.column_stack([u, v, n+endpoint[:, None]*d+np.arange(d)])
            values = np.column_stack([np.full(m, sign), np.full(m, -sign), -sign*delta])
            add(columns, values, .5*distance**2)
    if not cursor:
        return csr_matrix((0, n*(d+1))), np.empty(0)
    matrix = coo_matrix((np.concatenate(coefficients),
                         (np.concatenate(row_chunks), np.concatenate(col_chunks))),
                        shape=(cursor, n*(d+1))).tocsr()
    return matrix, np.concatenate(bounds)


def solve_coordinate(arguments):
    """Solve one coordinate LP, checking feasibility and the duality gap."""
    matrix, bound, score, weights, coordinate, dimension, time_limit = arguments
    n = len(weights)
    objective = np.zeros(n*(dimension+1))
    objective[:n] = -weights*score
    objective[n+np.arange(n)*dimension+coordinate] = -weights
    # SciPy forwards this HiGHS option even in versions that warn about it.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads',
                                category=OptimizeWarning)
        result = linprog(objective, A_ub=matrix, b_ub=bound,
            bounds=(-1., 1.), method='highs-ipm',
            options=dict(time_limit=time_limit, threads=1, primal_feasibility_tolerance=1e-8,
                         dual_feasibility_tolerance=1e-8, ipm_optimality_tolerance=1e-9))
    if not result.success:
        raise ValueError(f'Coordinate {coordinate} LP failed: {result.message}')
    violation = max(0., float(np.max(matrix@result.x-bound)) if len(bound) else 0.,
                    float(abs(result.x).max()-1))
    dual = (bound@result.ineqlin.marginals-result.lower.marginals.sum()
            +result.upper.marginals.sum())
    gap = abs(float(result.fun-dual))
    if violation > 1e-7 or gap > 1e-5*(1+abs(result.fun)):
        raise ValueError(f'Unresolved LP: violation={violation}, duality gap={gap}')
    return dict(coordinate=coordinate, discrepancy=float(-result.fun),
                iterations=int(result.nit), maximum_violation=violation,
                duality_gap=gap)


def graph_stein_discrepancy(points, scores, weights=None, neighbours=12,
                            stretch=2., executor=None, time_limit=300.):
    points, scores = np.asarray(points, dtype=float), np.asarray(scores, dtype=float)
    if points.ndim != 2 or scores.shape != points.shape or not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('Finite matching point and score matrices are required')
    n, d = points.shape
    weights = np.full(n, 1/n) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError('Weights must be a probability vector')
    edges, graph = verified_spanner(points, neighbours, stretch)
    matrix, bound = graph_constraints(points, edges)
    jobs = [(matrix, bound, scores[:, j], weights, j, d, time_limit) for j in range(d)]
    results = list(map(solve_coordinate, jobs) if executor is None else executor.map(solve_coordinate, jobs))
    return dict(discrepancy=sum(x['discrepancy'] for x in results),
                coordinates=results, support_points=n, constraints=matrix.shape[0],
                variables_per_coordinate=matrix.shape[1], graph=graph,
                norm='l1', stein_factors=[1., 1., 1.])
