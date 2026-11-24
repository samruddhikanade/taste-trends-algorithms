"""Tests for MODULE E (recommendation generation)."""
from __future__ import annotations

import numpy as np
import pytest

from src import metrics as M
from src.recommender_core import (FactorModel, popularity_ranking, recommend_popular, recommend_random,
                                  train_als_numpy)


@pytest.fixture
def small_model() -> FactorModel:
    users = np.array([10, 20])
    uf = np.array([[1.0, 0.0], [0.0, 1.0]])
    items = np.array([100, 200, 300, 400])
    itf = np.array([[5.0, 0.0], [4.0, 1.0], [1.0, 4.0], [0.0, 5.0]])
    return FactorModel(users, uf, items, itf, reg_param=0.1)


def test_recommend_orders_by_score_and_excludes_seen(small_model):
    recs = small_model.recommend(10, k=3)
    assert recs["movieId"].tolist() == [100, 200, 300]
    assert recs["rank"].tolist() == [1, 2, 3]
    assert np.allclose(recs["score"], [5.0, 4.0, 1.0])
    recs = small_model.recommend(10, k=3, exclude=[100])
    assert 100 not in recs["movieId"].tolist()
    assert recs["movieId"].tolist() == [200, 300, 400]


def test_recommend_never_returns_more_than_unseen_items(small_model):
    recs = small_model.recommend(20, k=10, exclude=[100, 200, 300])
    assert recs["movieId"].tolist() == [400]


def test_batch_matches_single_user(small_model):
    batch = small_model.recommend_batch(np.array([10, 20]), 2, {10: [100], 20: []})
    assert batch[10].tolist() == small_model.recommend(10, 2, exclude=[100])["movieId"].tolist()
    assert batch[20].tolist() == [400, 300]


def test_predict_and_cold_start(small_model):
    assert small_model.predict(10, 200) == pytest.approx(4.0)
    assert small_model.predict(999, 200) is None        # unseen user
    assert small_model.predict(10, 999) is None         # unseen movie
    with pytest.raises(KeyError):
        small_model.recommend(999)


def test_fold_in_recovers_user_preferences(small_model):
    # A new user who loves movies 100/200 and dislikes 400 should be close to user 10.
    vec = small_model.fold_in({100: 5.0, 200: 4.5, 400: 0.5})
    recs = small_model.recommend_vector(vec, k=1, exclude=[100, 200, 400])
    assert recs["movieId"].tolist() == [300]
    assert vec[0] > vec[1]
    with pytest.raises(ValueError):
        small_model.fold_in({12345: 5.0})


def test_similar_items(small_model):
    sims = small_model.similar_items(100, k=2)
    assert sims["movieId"].tolist() == [200, 300]


def test_popularity_and_random_baselines():
    ranking = popularity_ranking(np.array([1, 2, 3, 4]), np.array([10, 50, 50, 5]))
    assert ranking.tolist() == [2, 3, 1, 4]                   # ties broken by id
    assert recommend_popular(ranking, 2, exclude=[2]).tolist() == [3, 1]
    rng = np.random.default_rng(0)
    picks = recommend_random(np.arange(1, 11), 5, exclude=[1, 2], rng=rng)
    assert len(picks) == 5 and len(set(picks.tolist())) == 5 and not {1, 2} & set(picks.tolist())


def test_numpy_als_learns_low_rank_structure():
    rng = np.random.default_rng(42)
    n_users, n_items, rank = 60, 40, 3
    true_u = rng.normal(0, 1, (n_users, rank))
    true_i = rng.normal(0, 1, (n_items, rank))
    full = true_u @ true_i.T
    mask = rng.random(full.shape) < 0.5
    users, items = np.nonzero(mask)
    ratings = full[users, items]
    model = train_als_numpy(users, items, ratings, rank=3, reg_param=0.01, iterations=15, seed=1)
    train_pred = [model.predict(u, i) for u, i in zip(users, items)]
    hold_u, hold_i = np.nonzero(~mask)
    hold_pred = [model.predict(u, i) for u, i in zip(hold_u, hold_i)]
    assert M.rmse(train_pred, ratings) < 0.1
    # generalises to unseen cells much better than predicting the mean
    assert M.rmse(hold_pred, full[hold_u, hold_i]) < 0.5 * np.std(full[hold_u, hold_i])


def test_spark_als_trains_and_excludes_rated_movies(spark):
    from src.config import load_config
    from src.recommendation import build_als, factors_to_pandas, recommend_all_users

    rng = np.random.default_rng(3)
    rows = [(int(u), int(i), float(rng.integers(1, 11) / 2)) for u in range(1, 31) for i in range(1, 41)
            if rng.random() < 0.4]
    df = spark.createDataFrame(rows, "userId INT, movieId INT, rating FLOAT")
    cfg = load_config("dev")
    model = build_als(4, 0.1, 5, 42, cfg).fit(df)
    recs = recommend_all_users(model, df, k=10, candidate_k=40).toPandas()
    rated = {(u, i) for u, i, _ in rows}
    assert not any((int(r.userId), int(r.movieId)) in rated for r in recs.itertuples())
    assert recs.groupby("userId")["rank"].max().max() <= 10
    assert set(factors_to_pandas(model.itemFactors, "movieId").columns) == {"movieId", "f0", "f1", "f2", "f3"}
