# Methodology

## 1. Data

**MovieLens 25M** (GroupLens Research, University of Minnesota) contains 25,000,095 ratings and 1,093,360 tag applications. They cover 62,423 movies rated by 162,541 users between January 1995 and November 2019. Every user in the dataset rated at least 20 movies. Ratings use a 0.5-5.0 star scale in half-star steps.

| File | Columns | Used for |
|---|---|---|
| `ratings.csv` | userId, movieId, rating, timestamp | Everything: EDA, ALS training, evaluation |
| `movies.csv` | movieId, title, genres | Titles, release year, genre analysis |
| `tags.csv` | userId, movieId, tag, timestamp | Tag statistics, top tags per movie |
| `links.csv` | movieId, imdbId, tmdbId | Links to IMDb / TMDB in the dashboard |

**Dev mode** uses *ml-latest-small* (about 100k ratings from about 600 users, same file layout) so the whole pipeline can be developed and tested in minutes.

## 2. Ingestion (Module A)

* CSVs are read by Spark with **explicit schemas** and `PERMISSIVE` mode, so malformed values become nulls instead of crashing the job.
* The ingestion report records, for each file, the schema, row count, number of partitions, missing values per column and duplicates on the natural key. It also gives distinct users, movies and ratings. See `reports/results/<mode>/ingestion_report.json`.
* Optional **user sampling** (`--sample 0.2`) keeps a deterministic 20% of users with *all* their ratings. Sampling by user (`xxhash64(userId) mod 10000`) rather than by rating keeps each user's history complete.

## 3. Preprocessing (Module B)

| Rule | Implementation |
|---|---|
| Missing keys | drop rows with null userId / movieId / rating / timestamp |
| Invalid values | keep 0.5 ≤ rating ≤ 5.0 on the half-star scale, timestamp > 0, positive ids |
| Duplicates | for a repeated (userId, movieId), keep the most recent rating (`row_number()` over a window) |
| Referential integrity | drop ratings for movieIds missing from movies.csv (`left_semi` join with a **broadcast** movies table) |
| Timestamps | Unix seconds → `rating_ts` (UTC), `rating_year`, `rating_month` |
| Titles | regex `\((\d{4})\)\s*$` → `year`; title without the year → `clean_title` |
| Genres | `(no genres listed)` → empty; split on `|` into `genres_array` |
| Tags | lower-case, trim, drop empty |
| ML table | `(userId INT, movieId INT, rating FLOAT)`, `repartition(n, "userId")`, `cache()` |

Each rule's effect is counted and saved in `preprocessing_report.json`.

## 4. Big Data processing and EDA (Modules C and D)

All analytics are Spark SQL queries on temporary views (`ratings`, `movies`, `tags`, `links`, `movie_stats`):

* **Global summary:** totals, average rating, matrix density and sparsity, date range.
* **Rating distribution:** count and percentage per star value.
* **Movie popularity:** per-movie count, mean, standard deviation and Bayesian **weighted rating**: WR = v/(v+m)·R + m/(v+m)·C. Here v is the number of ratings, R the movie mean, C the global mean and m the minimum-count threshold (1,000 in full mode, 50 in dev).
* **Popularity tiers:** movies are ordered by rating count. **Head** = the most popular movies that together hold the first 50% of ratings, **Mid** = the next 30%, **Tail** = the rest. The rule is computed with window functions (`SUM(...) OVER (ORDER BY ...)`).
* **Long-tail curve:** cumulative share of ratings against the share of movies, using `NTILE`.
* **Genres:** `LATERAL VIEW explode(genres_array)`, then ratings, movies and rating-weighted average per genre, plus each genre's share by release decade.
* **Release years and time trends:** movies and average rating by release year; ratings, average rating, active users and movies rated per rating year; weekday × hour activity.
* **User behaviour:** ratings per user (activity buckets, median, 90th percentile via `percentile_approx`), histogram of users' average ratings, share of "generous" (average ≥ 4) and "harsh" (average < 3) users.

**Partitioning:** the ML table is hash-partitioned by `userId`, and its partition sizes are reported (`partition_report`). Adaptive Query Execution coalesces small shuffle partitions, and the small `movies` table is broadcast to avoid shuffling the large ratings table.

## 5. Recommendation engine (Module E)

1. **Split:** `randomSplit([0.8, 0.2], seed=42)` on individual ratings gives train and test sets.
2. **Hyper-parameter search:** the training set is split again 80/20 into fit/validation sets. In full mode only 10% of users are used for speed. A grid over `rank` (latent factors) × `regParam` (λ) is evaluated by validation RMSE, and the lowest RMSE wins. The grid is saved in `tuning_results.csv`.
3. **Final model:** Spark MLlib `ALS(rank, regParam, maxIter, coldStartStrategy="drop")` is trained on the full training set.
4. **Top-10 for every user:** `recommendForAllUsers(150)` scores 150 candidates per user with blocked matrix multiplication. A distributed **left anti-join** removes every movie the user has already rated (train *and* test), and `row_number()` keeps the best 10. Users who rated so many movies that fewer than 10 candidates remain are handled exactly at serving time (their rated lists are saved).
5. **Export:** user and item factor matrices, all top-10 lists, a movie catalogue and the rating histories of 300 demo users.
6. **Serving and cold start** (`recommender_core.py`, used by the dashboard):
   * *known user* → precomputed list, or live scoring `Q·pᵤ` with rated movies masked out;
   * *unknown user ID* → most highly rated popular movies (weighted rating, minimum-count threshold);
   * *new user who rates a few movies* → **fold-in**: pᵤ = (QᵀQ + λnI)⁻¹Qᵀr with Q fixed, then normal scoring. This is a single ALS user-step and needs no retraining;
   * *new movie with no ratings* → it has no factors and cannot be recommended by CF (a known limitation).

## 6. Evaluation (Module F)

All metrics are computed from the trained model's outputs on the held-out test set. Nothing is entered by hand.

**Cold-start filtering.** Test ratings whose user or movie never appears in training cannot be scored by any CF model. They are removed (and counted) **for every model**, so all RMSEs use identical rows.

| Metric | Definition | Plain-language meaning |
|---|---|---|
| RMSE | √(mean((r̂ − r)²)) over test ratings | Typical size of the error, in stars |
| Baselines | global mean μ; μ + bᵤ + bᵢ (damped, from the paper) | Does ALS beat "just predict the average"? |
| Precision@10 | hits in top-10 ÷ 10 | Share of recommendations the user really liked |
| Recall@10 | hits in top-10 ÷ number of liked test movies | Share of the user's liked movies that we found |
| NDCG@10 | position-discounted hits ÷ ideal | Hits near the top count more |
| HitRate@10 | users with ≥ 1 hit ÷ users | How often the list contains something good |
| Catalogue coverage | distinct recommended movies ÷ movies in training | How much of the catalogue is ever shown |
| ARP | mean training-rating count of recommended movies | How "mainstream" the recommendations are |
| Head/Mid/Tail share | share of recommendation slots per tier | Where the recommendations come from |
| Gini of exposure | inequality of how often each movie is shown | 0 = equal exposure, 1 = one movie for everyone |
| Popularity lift | ARP(recommendations) ÷ ARP(users' own histories) | > 1 means the system pushes more popular movies than users choose themselves |

**Ranking protocol.** "Relevant" means a test rating of 4.0 or higher. Up to 5,000 test users (1,000 in dev) are sampled with a fixed seed. Each model recommends 10 movies the user has **not rated in training**, chosen from movies that appear in training. Three recommenders are compared on exactly the same users:

* **ALS:** the highest predicted ratings,
* **Popularity:** the most-rated movies (the same list for everyone, minus what the user has seen),
* **Random:** a lower bound.

Coverage and bias are also reported for the stored top-10 lists of **all** users.

**Why precision values look small.** Each user has rated only a tiny fraction of the catalogue, and only 20% of those ratings are hidden. A recommended movie that the user would love, but never rated, counts as a miss. Absolute values are therefore low for every method, so the **comparison between methods** is what matters.

## 7. Reproducibility

* A single seed (`project.random_seed = 42`) controls sampling, splits, ALS initialisation and evaluation sampling.
* Versions are pinned in `requirements.txt`.
* `reports/results/<mode>/run_info.json` stores the Spark version and the time taken by each stage.
