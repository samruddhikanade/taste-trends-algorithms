"""Pure NumPy recommendation logic (no Spark needed).

Spark MLlib learns the latent factors on the full dataset. Once training is
finished, the factors are small (users x rank and movies x rank), so serving
recommendations is just a matrix product - which is what this module does.
It is used by the evaluation step, the Streamlit dashboard and the tests.

It also contains ``train_als_numpy``: a compact, from-scratch implementation of
the same ALS-WR algorithm that Spark uses. It mirrors the equations in Koren,
Bell & Volinsky (2009) and is used for unit tests and teaching - the real
training on 25M ratings is done by Spark (``recommendation.py``).

Model:  r̂(u, i) = p_u · q_i         (p_u = user factors, q_i = item factors)
Loss :  Σ (r_ui − p_u·q_i)² + λ (n_u ||p_u||² + n_i ||q_i||²)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.io_utils import read_table


@dataclass
class FactorModel:
    """Latent-factor model: ``user_ids``/``item_ids`` give the row of each id."""

    user_ids: np.ndarray
    user_factors: np.ndarray
    item_ids: np.ndarray
    item_factors: np.ndarray
    reg_param: float = 0.1
    _user_pos: dict[int, int] = field(init=False, repr=False)
    _item_pos: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.user_ids = np.asarray(self.user_ids, dtype=np.int64)
        self.item_ids = np.asarray(self.item_ids, dtype=np.int64)
        self.user_factors = np.asarray(self.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(self.item_factors, dtype=np.float32)
        if self.user_factors.shape[0] != len(self.user_ids) or self.item_factors.shape[0] != len(self.item_ids):
            raise ValueError("ids and factor matrices must have the same number of rows")
        self._user_pos = {int(u): i for i, u in enumerate(self.user_ids)}
        self._item_pos = {int(m): i for i, m in enumerate(self.item_ids)}

    @property
    def rank(self) -> int:
        return int(self.item_factors.shape[1])

    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_pos

    def has_item(self, item_id: int) -> bool:
        return int(item_id) in self._item_pos

    def item_positions(self, item_ids: Iterable[int]) -> np.ndarray:
        return np.array([self._item_pos[int(i)] for i in item_ids if int(i) in self._item_pos], dtype=np.int64)

    def user_vector(self, user_id: int) -> np.ndarray:
        return self.user_factors[self._user_pos[int(user_id)]]

    def predict(self, user_id: int, item_id: int) -> float | None:
        if not (self.has_user(user_id) and self.has_item(item_id)):
            return None  # cold start: no factors learned for this user or movie
        return float(self.user_vector(user_id) @ self.item_factors[self._item_pos[int(item_id)]])

    # ------------------------------------------------------------------ recommend
    def recommend_vector(self, vector: np.ndarray, k: int = 10, exclude: Iterable[int] = ()) -> pd.DataFrame:
        scores = self.item_factors @ np.asarray(vector, dtype=np.float32)
        return top_k_from_scores(scores, self.item_ids, k, self.item_positions(exclude))

    def recommend(self, user_id: int, k: int = 10, exclude: Iterable[int] = ()) -> pd.DataFrame:
        """Top-k unseen movies for a known user: columns ``movieId``, ``score``, ``rank``."""
        if not self.has_user(user_id):
            raise KeyError(f"user {user_id} has no learned factors (cold start)")
        return self.recommend_vector(self.user_vector(user_id), k, exclude)

    def recommend_batch(self, user_ids: np.ndarray, k: int, exclude: Mapping[int, Iterable[int]],
                        batch_size: int = 512) -> dict[int, np.ndarray]:
        """Top-k item ids for many users at once (vectorised, in memory-friendly batches)."""
        result: dict[int, np.ndarray] = {}
        user_ids = np.asarray(user_ids, dtype=np.int64)
        for start in range(0, len(user_ids), batch_size):
            batch = user_ids[start:start + batch_size]
            rows = np.array([self._user_pos[int(u)] for u in batch])
            scores = self.user_factors[rows] @ self.item_factors.T          # (batch, n_items)
            for r, u in enumerate(batch):
                seen = self.item_positions(exclude.get(int(u), ()))
                if len(seen):
                    scores[r, seen] = -np.inf
            top = _top_k_indices(scores, k)
            for r, u in enumerate(batch):
                valid = top[r][np.isfinite(scores[r, top[r]])]
                result[int(u)] = self.item_ids[valid]
        return result

    # ------------------------------------------------------------------ cold start
    def fold_in(self, ratings: Mapping[int, float]) -> np.ndarray:
        """Learn a vector for a NEW user from a few ratings, keeping movie factors fixed.

        This is exactly one ALS user-step:  p = (QᵀQ + λ·n·I)⁻¹ Qᵀ r
        """
        known = [(int(m), float(r)) for m, r in ratings.items() if self.has_item(m)]
        if not known:
            raise ValueError("none of the rated movies are known to the model")
        q = self.item_factors[[self._item_pos[m] for m, _ in known]].astype(np.float64)
        r = np.array([v for _, v in known])
        a = q.T @ q + self.reg_param * len(known) * np.eye(self.rank)
        return np.linalg.solve(a, q.T @ r).astype(np.float32)

    def similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        """Movies whose latent vectors point in the most similar direction (cosine)."""
        if not self.has_item(item_id):
            raise KeyError(f"movie {item_id} has no learned factors")
        norms = np.linalg.norm(self.item_factors, axis=1) + 1e-9
        target = self.item_factors[self._item_pos[int(item_id)]]
        sims = (self.item_factors @ target) / (norms * (np.linalg.norm(target) + 1e-9))
        return top_k_from_scores(sims, self.item_ids, k, np.array([self._item_pos[int(item_id)]]))


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values per row, sorted descending (works for 1-D or 2-D)."""
    k = min(k, scores.shape[-1])
    part = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    order = np.take_along_axis(scores, part, axis=-1)
    idx = np.argsort(-order, axis=-1, kind="stable")
    return np.take_along_axis(part, idx, axis=-1)


def top_k_from_scores(scores: np.ndarray, item_ids: np.ndarray, k: int, exclude_pos: np.ndarray) -> pd.DataFrame:
    scores = np.array(scores, dtype=np.float64, copy=True)
    if len(exclude_pos):
        scores[exclude_pos] = -np.inf
    top = _top_k_indices(scores, k)
    top = top[np.isfinite(scores[top])]
    return pd.DataFrame({"movieId": item_ids[top], "score": scores[top], "rank": np.arange(1, len(top) + 1)})


# ---------------------------------------------------------------------- baselines
def popularity_ranking(item_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Item ids sorted from most to least rated (ties broken by id for determinism)."""
    order = np.lexsort((np.asarray(item_ids), -np.asarray(counts)))
    return np.asarray(item_ids)[order]


def recommend_popular(ranking: np.ndarray, k: int, exclude: Iterable[int] = ()) -> np.ndarray:
    """First k items of a popularity ranking that the user has not seen."""
    seen = set(int(x) for x in exclude)
    out = []
    for item in ranking:
        if int(item) not in seen:
            out.append(int(item))
            if len(out) == k:
                break
    return np.array(out, dtype=np.int64)


def recommend_random(item_ids: np.ndarray, k: int, exclude: Iterable[int], rng: np.random.Generator) -> np.ndarray:
    seen = set(int(x) for x in exclude)
    picks: list[int] = []
    while len(picks) < k:
        cand = int(item_ids[rng.integers(0, len(item_ids))])
        if cand not in seen and cand not in picks:
            picks.append(cand)
        if len(picks) + len(seen) >= len(item_ids):
            break
    return np.array(picks, dtype=np.int64)


# ---------------------------------------------------------------- reference ALS
def train_als_numpy(users: np.ndarray, items: np.ndarray, ratings: np.ndarray, rank: int = 10,
                    reg_param: float = 0.1, iterations: int = 10, seed: int = 42,
                    verbose: bool = False) -> FactorModel:
    """Explicit-feedback ALS with weighted-λ regularisation (ALS-WR), as in Spark MLlib.

    Alternates between solving every user vector with item vectors fixed and
    every item vector with user vectors fixed. Each step is a small ridge
    regression with a closed-form solution - that is the "least squares" in ALS.
    Intended for small data (tests, the dev dataset); Spark is used for 25M.
    """
    users, items = np.asarray(users), np.asarray(items)
    ratings = np.asarray(ratings, dtype=np.float64)
    u_ids, u_idx = np.unique(users, return_inverse=True)
    i_ids, i_idx = np.unique(items, return_inverse=True)
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.1, (len(u_ids), rank))
    q = rng.normal(0, 0.1, (len(i_ids), rank))
    by_user = _group(u_idx, len(u_ids))
    by_item = _group(i_idx, len(i_ids))
    eye = np.eye(rank)

    def solve(groups: list[np.ndarray], other_idx: np.ndarray, other: np.ndarray, target: np.ndarray) -> None:
        for row, obs in enumerate(groups):
            if len(obs) == 0:
                continue
            f = other[other_idx[obs]]
            target[row] = np.linalg.solve(f.T @ f + reg_param * len(obs) * eye, f.T @ ratings[obs])

    for it in range(iterations):
        solve(by_user, i_idx, q, p)
        solve(by_item, u_idx, p, q)
        if verbose:
            pred = np.sum(p[u_idx] * q[i_idx], axis=1)
            print(f"  iter {it + 1}: train RMSE = {np.sqrt(np.mean((pred - ratings) ** 2)):.4f}")
    return FactorModel(u_ids, p, i_ids, q, reg_param=reg_param)


def _group(index: np.ndarray, n: int) -> list[np.ndarray]:
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(n + 1))
    return [order[bounds[g]:bounds[g + 1]] for g in range(n)]


def load_factor_model(processed_dir: str | Path, reg_param: float = 0.1) -> FactorModel | None:
    """Load the factors exported by the Spark training step (``None`` if not trained yet)."""
    users = read_table(Path(processed_dir) / "user_factors")
    items = read_table(Path(processed_dir) / "item_factors")
    if users is None or items is None:
        return None
    fcols = [c for c in users.columns if c.startswith("f") and c[1:].isdigit()]
    return FactorModel(users["userId"].values, users[fcols].values, items["movieId"].values, items[fcols].values,
                       reg_param=reg_param)
