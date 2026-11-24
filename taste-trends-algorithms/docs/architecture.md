# System Architecture

## 1. Overview

The system has three layers:

1. **Big Data processing (offline, PySpark).** Spark runs in *local mode* on a single laptop. It ingests the 25M-row MovieLens dataset, cleans it, runs Spark SQL analytics, trains the ALS model with Spark MLlib, generates top-10 lists for every user and evaluates the model.
2. **Results store (files).** Spark writes only small, aggregated outputs: CSV/JSON results, the latent-factor matrices and the recommendation lists. The full ratings table is never converted to pandas.
3. **Presentation (online, Streamlit + Plotly).** The dashboard reads those files and serves recommendations with NumPy dot products. It never starts Spark, so it opens in seconds.

```mermaid
flowchart LR
    subgraph SRC["Data source"]
        ML[("MovieLens 25M<br/>ratings.csv · movies.csv<br/>tags.csv · links.csv")]
    end

    subgraph SPARK["Apache Spark (local mode, all CPU cores)"]
        direction TB
        A["A · Ingestion<br/>explicit schemas, quality report"]
        B["B · Preprocessing<br/>clean · dedupe · timestamps · genres"]
        P["Partitioned ML table<br/>(userId, movieId, rating)<br/>hash-partitioned by userId, cached"]
        CD["C/D · Spark SQL analytics<br/>popularity · genres · trends · users"]
        E["E · MLlib ALS<br/>grid search → final model<br/>recommendForAllUsers + anti-join"]
        F["F · Evaluation<br/>RMSE · P@10 · R@10 · NDCG<br/>coverage · popularity bias"]
        A --> B --> P
        B --> CD
        P --> E --> F
    end

    subgraph STORE["Results store (small files)"]
        R1["reports/results/&lt;mode&gt;/<br/>CSV + JSON + RESULTS.md"]
        R2["data/processed/&lt;mode&gt;/<br/>factors · top-10 lists · catalogue"]
        R3["reports/figures/&lt;mode&gt;/<br/>interactive HTML charts"]
    end

    subgraph UI["Streamlit dashboard (app.py)"]
        D1["Overview · Dataset · Popularity · Genres"]
        D2["Personalised recommendations<br/>(NumPy scoring + cold-start fold-in)"]
        D3["Model performance · ALS vs Popularity"]
    end

    ML --> A
    CD --> R1
    CD --> R3
    E --> R2
    F --> R1
    R1 --> D1
    R1 --> D3
    R2 --> D2
```

## 2. Data flow, step by step

```mermaid
sequenceDiagram
    participant U as User (PowerShell)
    participant DL as download_data.py
    participant SP as Spark driver + executors
    participant FS as Disk (results)
    participant ST as Streamlit

    U->>DL: python scripts/download_data.py --mode full
    DL->>FS: data/raw/ml-25m/*.csv (checksum verified)
    U->>SP: python scripts/run_pipeline.py --mode full
    SP->>SP: read CSV with schemas → cache → profile
    SP->>SP: clean → partition by userId → cache
    SP->>FS: EDA tables (Spark SQL → toPandas on small results)
    SP->>SP: randomSplit 80/20 → ALS grid search → final ALS
    SP->>SP: recommendForAllUsers(150) → left_anti join rated → top-10
    SP->>FS: factors, top-10 lists, metrics.json, RESULTS.md
    U->>ST: streamlit run app.py
    ST->>FS: read results (cached)
    ST-->>U: interactive dashboard
```

## 3. Why this design

| Decision | Reason |
|---|---|
| Spark **local mode** (`local[*]`) | Uses every CPU core as a parallel worker without needing a Hadoop/YARN cluster. The same code runs unchanged on a real cluster by changing `spark.master`. |
| **Explicit schemas** for CSV | Avoids a full extra pass over 25M rows for `inferSchema`, keeps `imdbId` as a string (leading zeros) and turns malformed values into nulls we can count. |
| **Hash-partitioning by `userId`** + `cache()` | All ratings of one user sit in the same partition, which reduces shuffling for per-user work; caching avoids re-reading the CSV for each of ALS's iterations. |
| **Adaptive Query Execution** | Spark merges tiny shuffle partitions at runtime, so a single `shuffle_partitions` setting works for both the 100k and the 25M dataset. |
| **Spark SQL** for analytics | Declarative queries that read like the report; Catalyst optimises them (e.g. broadcast join with the small movies table). |
| **No Spark writes** (pandas saves the small results) | On Windows, Spark's Hadoop writer needs `winutils.exe`. Saving only aggregated outputs with pandas avoids that and keeps the project runnable on any laptop. |
| **Factors exported for serving** | Predicting is just `p_u · q_i`. Exporting P and Q lets the dashboard recommend in milliseconds and handle **cold-start** users by "folding in" a new user vector. |
| **Dev and full modes** | Dev mode (ml-latest-small, about 100k ratings) runs in minutes for development and testing; full mode processes all 25M ratings. Outputs are kept in separate folders. |

## 4. Module map

| Layer | File | Responsibility |
|---|---|---|
| Config | `src/config.py`, `config/config.example.yaml` | Modes, paths, Spark memory, ALS grid |
| Spark | `src/spark_session.py` | Local Spark session tuned for a laptop |
| A | `src/ingestion.py` | CSV → DataFrames, schemas, counts, missing values, duplicates |
| B | `src/preprocessing.py` | Cleaning rules, timestamps, year/genre parsing, ML table, split |
| C | `src/eda.py` | Runs the analytics, saves tables, exports Plotly HTML |
| D | `src/analytics.py` | Spark SQL queries: popularity, tiers, genres, trends, user behaviour |
| E | `src/recommendation.py` | ALS tuning/training, top-10 for all users, export |
| E | `src/recommender_core.py` | NumPy serving: scoring, exclusion, cold-start fold-in, similar movies, reference ALS |
| F | `src/evaluation.py`, `src/metrics.py` | RMSE, baselines, ranking metrics, coverage, popularity bias |
| Report | `src/report.py` | `RESULTS.md` and README results section from the real outputs |
| UI | `app.py`, `dashboard/*.py`, `src/visuals.py` | Streamlit pages, theme, Plotly charts |
