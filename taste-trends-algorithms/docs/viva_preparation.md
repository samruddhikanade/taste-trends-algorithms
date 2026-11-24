# BDA Viva Preparation - Taste, Trends & Algorithms

*Short answers you can say out loud. Once your own run has finished, fill the blanks [ ] with the numbers from `reports/results/full/RESULTS.md`.*

---

## 1. Project in 30 seconds

> "I built a movie recommender on the 25-million-rating MovieLens dataset with Apache Spark. Spark SQL analyses user behaviour, genres and popularity. Spark MLlib's ALS algorithm, based on Koren, Bell and Volinsky's matrix factorisation paper, predicts ratings and recommends 10 unseen movies per user. I then measured accuracy with RMSE and Precision/Recall@10, and checked for popularity bias: whether ALS mostly recommends already-popular movies compared with a simple most-popular list. A Streamlit dashboard lets you pick a user and see their recommendations."

## 2. Important technical terms

| Term | One-line meaning |
|---|---|
| **Big Data (5 Vs)** | Volume, Velocity, Variety, Veracity, Value: data too large or complex for one-machine tools. |
| **Apache Spark** | In-memory, distributed data-processing engine: the same code runs on one laptop or on 1,000 machines. |
| **Driver / Executor** | The driver plans the job; executors run tasks on partitions. In local mode both live in one JVM. |
| **Local mode `local[*]`** | Spark uses every CPU core of one machine as a worker. No cluster is needed. |
| **RDD** | Resilient Distributed Dataset: Spark's low-level, fault-tolerant, partitioned collection. |
| **DataFrame** | A distributed table with a schema, built on RDDs and optimised by Catalyst. |
| **Spark SQL / Catalyst** | Run SQL on DataFrames. Catalyst rewrites the query plan, for example pushing filters down or choosing a broadcast join. |
| **Partition** | A chunk of the data processed by one task. More partitions means more parallelism. |
| **Shuffle** | Moving data between partitions (for joins and groupBy). It is the most expensive operation. |
| **Transformation vs Action** | Transformations (`select`, `where`, `join`) are **lazy**. Actions (`count`, `collect`) trigger execution. |
| **Lazy evaluation / DAG** | Spark builds a graph of steps and runs it only when an action is called, so it can optimise the whole graph. |
| **Cache / persist** | Keep a DataFrame in memory so later steps (such as ALS iterations) don't recompute it. |
| **Broadcast join** | Send a small table to every executor so the big table is never shuffled. |
| **AQE** | Adaptive Query Execution: Spark re-optimises at runtime, for example merging tiny partitions. |
| **MLlib** | Spark's distributed machine-learning library (`pyspark.ml`). |
| **Collaborative filtering** | Recommend using the behaviour of similar users or items, not item content. |
| **Matrix factorisation** | Approximate the sparse rating matrix R ≈ P·Qᵀ with small user and item factor matrices. |
| **Latent factor** | A hidden dimension of taste learned from data (for example "light comedy ↔ dark drama"). |
| **Rank** | The number of latent factors per user or movie. |
| **Regularisation (λ, regParam)** | A penalty on large factor values that prevents overfitting. |
| **Cold start** | No history for a new user or new movie, so collaborative filtering cannot score it. |
| **Sparsity** | Share of user × movie cells with no rating. MovieLens 25M: about [ ]%. |
| **Popularity bias** | Recommenders over-expose already-popular items, so the rich get richer. |
| **Long tail** | Many niche movies that each get few ratings. |

## 3. Collaborative filtering

* **Idea:** "People who agreed in the past will agree in the future." It needs only the user × item rating matrix, with no genres or actors.
* **Two types:**
  * *Memory/neighbourhood-based:* find similar users or items (cosine or Pearson similarity) and average their ratings.
  * *Model-based:* learn a compact model of the matrix. **Matrix factorisation (ALS)** is what I used.
* **Advantages:** discovers non-obvious tastes and needs no metadata.
* **Disadvantages:** cold start, sparsity, popularity bias, and the system can't explain *why* in human terms.

## 4. ALS explained

1. Arrange ratings in a matrix R (users × movies). It is more than 99% empty.
2. We want R ≈ P·Qᵀ, where each user *u* gets a vector pᵤ and each movie *i* a vector qᵢ of length *rank*. A prediction is the dot product: r̂ᵤᵢ = pᵤ·qᵢ.
3. Minimise the squared error on **observed** ratings plus λ·(‖pᵤ‖² + ‖qᵢ‖²).
4. With both P and Q unknown the problem is hard, but **fix Q and each pᵤ is a simple least-squares (ridge regression) problem** with a closed-form answer: pᵤ = (QᵀQ + λnᵤI)⁻¹Qᵀrᵤ.
5. **Alternate:** fix Q and solve all users; fix P and solve all movies; repeat for `maxIter` rounds. The error goes down every round.
6. **Why it suits Spark:** every user vector in a round is independent, so Spark solves them **in parallel** across partitions (user and item "blocks").

**My parameters:** rank = [ ], regParam = [ ], maxIter = [ ], chosen by grid search on validation RMSE. `coldStartStrategy="drop"` skips users or movies unseen during training when scoring.

**ALS vs SGD:** SGD updates after each rating and is simple but sequential. ALS solves exact sub-problems and parallelises naturally.

## 5. Spark in this project

* `spark_session.py` creates `local[*]` with driver memory 6 GB (full) or 2 GB (dev), shuffle partitions, AQE, Arrow and Kryo.
* **Ingestion:** `spark.read.csv` with an explicit `StructType` schema, without `inferSchema`.
* **Preprocessing:** `dropna`, `where`, a `row_number()` window to remove duplicates, a broadcast `left_semi` join, and `timestamp_seconds`.
* **Partitioning:** `repartition(n, "userId")` + `cache()` for the ML table.
* **Spark SQL:** CTEs, window functions (`ROW_NUMBER`, cumulative `SUM OVER`, `NTILE`), `LATERAL VIEW explode`, `percentile_approx`.
* **MLlib:** `ALS`, `RegressionEvaluator`, `recommendForAllUsers`.
* **Only small results** (aggregates, factors, top-10 lists) are moved to pandas. The 25M rows never leave Spark.

## 6. RMSE

* RMSE = √( (1/N) Σ (predicted − actual)² ).
* Unit: stars. RMSE 0.80 means predictions are typically about 0.8 stars off.
* It penalises large errors more than MAE does, because errors are squared.
* **Compare against baselines:** global mean [ ], μ + bᵤ + bᵢ [ ], ALS [ ].
* **Limitation:** good RMSE ≠ good top-10 list. RMSE is computed on movies people chose to rate, not on what we would show them. That is why I also report ranking metrics.

## 7. Precision@10, Recall@10, coverage

* **Precision@10** = (liked movies in my top 10) ÷ 10. "Liked" means rated ≥ 4 in the hidden test set.
* **Recall@10** = (liked movies in my top 10) ÷ (all liked movies in the test set for that user).
* Values are small (for example 0.0x) because unrated good movies count as misses. **Compare methods, not absolute values.**
* **Catalogue coverage** = share of all movies that get recommended to anyone. The popularity baseline has tiny coverage because everyone gets the same list.

## 8. Popularity bias

* **Definition:** the recommender shows popular items more often than their share of real user interest, so niche items rarely get exposure. Exposure creates more ratings, which creates more exposure: a feedback loop.
* **How I measured it:**
  * **Head/Mid/Tail tiers:** Head = the most-rated movies that together hold 50% of all ratings.
  * **ARP** = average popularity of recommended movies.
  * **Popularity lift** = ARP(recommendations) ÷ ARP(what users rated themselves).
  * **Gini of exposure** and the Lorenz curve.
* **My finding:** ALS gave [ ]% of slots to Head movies, against [ ]% of all ratings; lift [ ]×; coverage [ ]% vs [ ]% for the popularity list.
* **Fixes (future scope):** re-ranking for diversity, inverse-propensity weighting, calibrated recommendations, exploration (bandits), and adding a novelty or serendipity objective.

## 9. Questions a professor may ask

1. **Why Spark and not pandas?** 25M rows (about 650 MB of CSV) plus joins, window functions and iterative ML would exhaust a single pandas process's memory. Spark processes partitions in parallel, spills to disk when needed and scales to a cluster without code changes.
2. **Is local mode really "Big Data"?** The code is fully distributed: tasks run per partition in parallel. Changing `spark.master` to a cluster URL runs the same program on many machines.
3. **What is a partition and how did you choose the number?** A chunk of data processed by one task. About 2-4 per CPU core. I used 48 shuffle partitions for 25M rows and 8 for dev. AQE merges small ones.
4. **What is lazy evaluation? Give an example from your code.** `clean_ratings` chains `dropna → where → window → join` without running anything. Spark runs it at `.count()`.
5. **Why explicit schemas?** They are faster (no inference pass), keep correct types (imdbId as a string keeps leading zeros) and make malformed values into nulls I can count.
6. **How did you handle duplicates and invalid data?** See the preprocessing report: [ ] missing, [ ] invalid, [ ] duplicates removed.
7. **Explain ALS in simple words.** See section 4: guess the movie vectors, solve the users, solve the movies, repeat.
8. **What do rank and regParam do? What happens if rank is too high?** Rank is model capacity. Too high a rank overfits (memorises) and is slower. λ counters overfitting.
9. **How did you tune hyperparameters?** A grid of rank × regParam evaluated by validation RMSE on a split of the training data. The test set is never used for tuning.
10. **What is the cold-start problem and how do you handle it?** New users get a popular fallback list or fold-in from a few ratings. New movies can't be recommended by CF; that needs content features (future scope).
11. **Why drop cold-start rows in evaluation?** No CF model can score them. Dropping them for all models keeps the comparison fair, and I report how many were dropped.
12. **Why exclude already-rated movies?** Recommending something the user has already seen is useless. I do it with a distributed left anti-join.
13. **Is a low precision@10 bad?** Not by itself. Missing labels make absolute numbers low. ALS vs popularity vs random is the meaningful comparison.
14. **Which is better, ALS or popularity?** Use your numbers. Accuracy: [ ]. Diversity/coverage: ALS [ ] vs [ ]. Popularity is a strong accuracy baseline because popular movies are what many people liked, but it gives everyone the same list.
15. **What is the difference between explicit and implicit feedback?** Explicit = star ratings (used here). Implicit = clicks or views, where Spark's ALS uses `implicitPrefs=True` with confidence weights.
16. **What would you change on a real cluster?** HDFS/S3 storage, Parquet instead of CSV, more partitions, YARN or Kubernetes, scheduled retraining, a serving API.
17. **How does the dashboard give recommendations so fast?** It doesn't run Spark. It uses the exported factors, so scoring is one matrix-vector product.
18. **What is the Bayesian weighted rating?** It shrinks a movie's mean towards the global mean when it has few ratings, so 3 ratings of 5★ don't beat 50,000 ratings at 4.4★.
19. **What does the long-tail chart show?** The top 1% of movies receive [ ]% of all ratings.
20. **What are the limitations?** A random split rather than a time-based one, only explicit ratings, no content features, popularity bias not yet mitigated, and only one machine.

## 10. Numbers to memorise (from your run)

| Fact | Value |
|---|---|
| Ratings / users / movies | [ ] / [ ] / [ ] |
| Sparsity | [ ]% |
| Average rating | [ ] |
| Best rank, regParam | [ ], [ ] |
| RMSE: ALS / bias baseline / mean | [ ] / [ ] / [ ] |
| Precision@10: ALS / popularity / random | [ ] / [ ] / [ ] |
| Coverage: ALS / popularity | [ ] / [ ] |
| Head share of ALS slots vs Head share of ratings | [ ] vs [ ] |
| Pipeline runtime (full) | [ ] min |
