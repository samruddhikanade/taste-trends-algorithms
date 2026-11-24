# Research Paper Notes

**Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer*, vol. 42, no. 8, pp. 30-37, 2009. DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)**

> These notes summarise the paper's ideas in my own words and map them to this project. The project is an **independent implementation** of the concepts using Apache Spark MLlib on MovieLens 25M. It does **not** reproduce the paper's Netflix Prize experiments or numbers.

## 1. Context

The authors were part of the team that won the Netflix Prize, an open competition to predict how users would rate movies. The Netflix data contained more than 100 million ratings from about 480,000 users on 17,770 movies. The paper explains why **matrix factorisation** models became the strongest single technique in that competition.

## 2. Two families of recommender systems

| Approach | Idea | Weakness |
|---|---|---|
| **Content filtering** | Build a profile of each item (genre, actors, ...) and each user, then match them. | Needs external information that may be hard to collect. |
| **Collaborative filtering (CF)** | Use only past behaviour (ratings). Users who agreed in the past tend to agree in the future. | **Cold start**: cannot handle new users or items without any history. |

Collaborative filtering itself has two main forms:

* **Neighbourhood methods** compute relationships between items (or users). For example, "people who liked *Saving Private Ryan* also liked other war films".
* **Latent factor models** explain ratings by describing items and users with a small number of inferred **factors**. For movies, a factor might capture comedy vs drama, the amount of action, or orientation towards children. Some factors have no clear human meaning.

## 3. The basic matrix factorisation model

Each item *i* gets a vector **qᵢ** and each user *u* a vector **pᵤ**, both of length *f* (the number of latent factors, Spark's `rank`). Each element of qᵢ says how strongly the item has a factor, and each element of pᵤ says how much the user likes that factor. The predicted rating is their dot product:

r̂ᵤᵢ = qᵢᵀ pᵤ

The rating matrix is extremely sparse. In MovieLens 25M more than 99% of user × movie cells are empty. So the model is fitted **only on observed ratings**, with **regularisation** to avoid overfitting:

min over p, q of  Σ₍ᵤ,ᵢ₎ ∈ observed  (rᵤᵢ − qᵢᵀpᵤ)²  +  λ (‖qᵢ‖² + ‖pᵤ‖²)

λ (Spark's `regParam`) controls how strongly large factor values are penalised. Its value is chosen by validation, which this project does with a grid search.

## 4. Learning algorithms

* **Stochastic gradient descent (SGD):** loop over ratings and nudge pᵤ and qᵢ in the direction that reduces the error.
* **Alternating Least Squares (ALS):** the objective is not convex when p and q are both unknown, but if all qᵢ are fixed it becomes an ordinary least-squares problem in pᵤ, and the reverse. ALS therefore **alternates**:
  1. fix item factors and solve every user vector exactly (a small ridge regression each),
  2. fix user factors and solve every item vector,
  3. repeat until convergence.

The paper notes that ALS is preferable when the system can use **parallelisation**, because each user (or item) vector is computed independently of the others, and for implicit-feedback data. This is why Spark MLlib implements ALS: users and items are split into blocks that are solved in parallel across the cluster.

*Implementation detail:* Spark uses the "ALS-WR" variant (Zhou et al., 2008). Each vector's penalty is multiplied by the number of ratings it has (λ·nᵤ·‖pᵤ‖²). This project's NumPy reference implementation (`train_als_numpy`) and the cold-start fold-in use the same formula:

pᵤ = (QᵢᵀQᵢ + λ nᵤ I)⁻¹ Qᵢᵀ rᵤ

## 5. Extensions discussed in the paper

| Extension | Idea | Use in this project |
|---|---|---|
| **Biases** | Much of the variation in ratings comes from user or item effects alone: some users rate high, some movies are rated high. Baseline bᵤᵢ = μ + bᵤ + bᵢ. | Implemented as a **baseline predictor** for the RMSE comparison (`evaluation.baseline_rmses`). Spark's ALS itself models only qᵢᵀpᵤ. |
| **Additional input sources** | Implicit feedback (what a user rated, not how) and user attributes help with cold start. | Not modelled. Cold start is handled by a popularity fallback and by fold-in from a few ratings. |
| **Temporal dynamics** | Item popularity, user rating scales and tastes drift over time. | Analysed descriptively (ratings over time, average rating by year), not modelled. |
| **Varying confidence** | Some observations deserve more weight. | Not used (explicit-rating ALS). |

## 6. Concept → code map

| Paper concept | Where it lives |
|---|---|
| Rating matrix R, sparsity | `analytics.global_summary` (density/sparsity), `preprocessing.prepare_ml_data` |
| Latent factors pᵤ, qᵢ, dimension f | `ALS(rank=...)`, exported as `user_factors`/`item_factors` |
| Regularised squared error, λ | `ALS(regParam=...)`, grid search in `recommendation.tune` |
| Alternating Least Squares | Spark MLlib `ALS`, and `recommender_core.train_als_numpy` for teaching |
| Prediction r̂ᵤᵢ = qᵢᵀpᵤ | `FactorModel.predict`, Spark `model.transform` |
| Baseline μ + bᵤ + bᵢ | `evaluation.baseline_rmses` |
| RMSE | `RegressionEvaluator(metricName="rmse")`, `metrics.rmse` |
| Cold start | `coldStartStrategy="drop"`, `FactorModel.fold_in`, popularity fallback in the dashboard |

## 7. What this project adds beyond the paper's focus

The paper evaluates models by **rating-prediction accuracy (RMSE)**. This project also asks whether accurate models produce **balanced recommendations**. It measures top-10 ranking quality (Precision/Recall/NDCG@10), **catalogue coverage** and **popularity bias** (ARP, Head/Mid/Tail exposure, Gini). It compares ALS against a most-popular baseline. Later research found that collaborative filtering tends to amplify popularity, for example Abdollahpouri et al. (2019) on managing popularity bias, and this project measures that effect on MovieLens 25M.

## 8. Other references

* F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context", *ACM TiiS* 5(4), 2015. Dataset citation required by GroupLens.
* Y. Zhou, D. Wilkinson, R. Schreiber and R. Pan, "Large-Scale Parallel Collaborative Filtering for the Netflix Prize", *AAIM* 2008. The ALS-WR algorithm used by Spark.
* H. Abdollahpouri, R. Burke and B. Mobasher, "Managing Popularity Bias in Recommender Systems with Personalized Re-ranking", *FLAIRS* 2019.
* Apache Spark MLlib documentation, "Collaborative Filtering": https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
