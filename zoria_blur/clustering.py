"""Cluster tracks into persons using ArcFace embeddings.

Average-linkage agglomerative clustering on track descriptors with a
cannot-link constraint: two tracks that are visible in the same frames are
never the same person.
"""
from __future__ import annotations

import numpy as np


def track_descriptor(embs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | None:
    if embs is None or len(embs) == 0:
        return None
    w = np.ones(len(embs)) if weights is None else np.asarray(weights, dtype=np.float64) + 1e-3
    d = (embs * w[:, None]).sum(0)
    return (d / (np.linalg.norm(d) + 1e-9)).astype(np.float32)


def _overlap(a: set, b: set, tol: int) -> bool:
    if not a or not b:
        return False
    return len(a & b) > tol


def cluster_tracks(descs: list, frame_sets: list[set], sim_thresh: float = 0.40,
                   overlap_tol: int = 2) -> list[int]:
    """Returns a cluster label per track (tracks without descriptor get their own label)."""
    n = len(descs)
    valid = [i for i in range(n) if descs[i] is not None]
    clusters: list[list[int]] = [[i] for i in valid]
    cframes: list[set] = [set(frame_sets[i]) for i in valid]
    if valid:
        D = np.stack([descs[i] for i in valid])
        S_full = D @ D.T
        pos = {t: k for k, t in enumerate(valid)}
    while len(clusters) > 1:
        best, bi, bj = sim_thresh, -1, -1
        for i in range(len(clusters)):
            ii = [pos[t] for t in clusters[i]]
            for j in range(i + 1, len(clusters)):
                jj = [pos[t] for t in clusters[j]]
                s = float(S_full[np.ix_(ii, jj)].mean())
                if s >= best and not _overlap(cframes[i], cframes[j], overlap_tol):
                    best, bi, bj = s, i, j
        if bi < 0:
            break
        clusters[bi] += clusters[bj]
        cframes[bi] |= cframes[bj]
        del clusters[bj], cframes[bj]
    labels = [-1] * n
    for c, members in enumerate(clusters):
        for t in members:
            labels[t] = c
    nxt = len(clusters)
    for i in range(n):
        if labels[i] < 0:
            labels[i] = nxt
            nxt += 1
    return labels
