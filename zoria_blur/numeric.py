"""Small numpy replacements for the two scipy routines the engine used (scipy added ~65 MB to the
Windows package for one assignment solver and one 1-D smoothing filter)."""
from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost):
    """Minimum-cost assignment, same contract as scipy.optimize.linear_sum_assignment:
    returns (row_ind, col_ind) with row_ind ascending and min(n, m) pairs. Finite costs only.
    Hungarian method with potentials (shortest augmenting path), O(n^2 m)."""
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    if c.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if not np.all(np.isfinite(c)):
        raise ValueError("cost matrix contains non-finite values")
    tr = c.shape[0] > c.shape[1]
    if tr:
        c = c.T
    n, m = c.shape                      # n <= m
    INF = np.inf
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)       # p[j] = row (1-based) assigned to column j, 0 = free
    way = np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0 - 1] - u[i0] - v[1:]            # reduced costs to every column
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            uj = np.nonzero(used)[0]
            u[p[uj]] += delta
            v[uj] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    if tr:
        rows, cols = cols, rows
    order = np.argsort(rows, kind="stable")
    return rows[order].astype(np.int64), cols[order].astype(np.int64)


def gaussian_filter1d_nearest(x, sigma, axis=0, truncate=4.0):
    """scipy.ndimage.gaussian_filter1d(x, sigma, axis, mode='nearest') for float input."""
    x = np.asarray(x, dtype=np.float64)
    r = int(truncate * float(sigma) + 0.5)
    t = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (t / sigma) ** 2)
    w /= w.sum()
    xm = np.moveaxis(x, axis, 0)
    pad = np.concatenate([np.repeat(xm[:1], r, 0), xm, np.repeat(xm[-1:], r, 0)], axis=0)
    out = np.zeros_like(xm)
    n = xm.shape[0]
    for k in range(2 * r + 1):
        out += w[k] * pad[k:k + n]
    return np.moveaxis(out, 0, axis)
