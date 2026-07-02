---
marp: true
title: "R^{2k} is Theoretically Large Enough for Embedding-based Top-k Retrieval"
description: "Five-minute ICML 2026 video recording deck source"
paginate: true
math: mathjax
---

<!--
Target: 5-minute ICML-style video, 7 slides.

Canonical editable deck:
- slides_beamer.tex

Primary sources:
- README.md
- paper/draft_main.tex
- results/unlimit/random_embeddings/limit_retrieval_table.tex
- results/unlimit/cyclic_overfit/limit_cyclic_overfit_summary.tex

Visual assets:
- ../../paper/figures/top2_dimension_fit.pdf
- ../../paper/figures/limit_retrieval_limit.pdf
- ../../paper/figures/limit_retrieval_limit_small.pdf
-->

<!-- _class: lead -->

# $\mathbb{R}^{2k}$ is theoretically large enough for embedding-based top-$k$ retrieval

Zihao Wang, Hang Yin, Lihui Liu, Hanghang Tong, Yangqiu Song, Ginny Wong, Simon See

**Main message:** exact geometric approximability is not the obstruction.

---

# Settings: exact and robust retrieval

Embedding retrieval stores $m$ objects as vectors and ranks them by query-object scores.

**Exact MED:** for every answer set

$$
S\subseteq X,\qquad 1\le |S|\le k,
$$

there is a query $u_S$ and threshold $b_S$ such that

$$
\min_{i\in S}s(v_i,u_S)>b_S\ge \max_{j\notin S}s(v_j,u_S).
$$

If $|S|$ is known, this is top-$|S|$ retrieval.

**Robust MED:** unit-normalized objects and queries must also satisfy

$$
\min_{i\in S}s(v_i,u_S)
\ge
\max_{j\notin S}s(v_j,u_S)+\epsilon.
$$

---

# MED

**Upper bound:** place objects on the moment curve in $\mathbb{R}^{2k}$:

$$
v_i=(t_i,t_i^2,\ldots,t_i^{2k}).
$$

For target $S$, define

$$
P_S(t)=\prod_{i\in S}(t-t_i),
$$

and use the coefficients of $-P_S(t)^2$ as the query vector. Selected objects score $0$; unselected objects score below $0$.

**Lower bound:** MED realizes every subset of any $k$ chosen objects, so the threshold class shatters $k$ points. Since linear thresholds in $\mathbb{R}^d$ have VC dimension $d+1$, we need $d\ge k-1$.

| Scoring rule | Lower bound | Upper bound |
|---|---:|---:|
| Inner product | $k-1$ | $2k$ |
| Euclidean distance | $k-1$ | $2k$ |
| Cosine similarity | $k-1$ | $2k+1$ |

---

# RMED: feasible margins and Gaussian upper bound

Robust retrieval is impossible above the finite-$m$ margin ceiling

$$
\epsilon_\star(m,k)=\frac{m}{\sqrt{k(m-1)(m-k)}}.
$$

The ceiling is attained by a regular simplex in $\mathbb{R}^{m-1}$.

In the retrieval regime $m/k\to\infty$,

$$
\epsilon_\star(m,k)\sim \frac{1}{\sqrt{k}}.
$$

At the feasible scale $\epsilon_k=c/\sqrt{k}$, a Gaussian centroid witness uses

$$
d=Ck^2\log m,\qquad
u_S=\frac{\sum_{i\in S}v_i}{\left\|\sum_{i\in S}v_i\right\|_2},
$$

and gives

$$
\operatorname{RMED\text{-}C}(m,k,\epsilon_k)\le O(k^2\log m).
$$

---

# Experiments (1): synthetic top-2 query

For $k=2$, the cyclic-polytope construction predicts an exact $d=4$ witness.

![height:360px](../../paper/figures/top2_dimension_fit.pdf)

- Cyclic polytope stays at $d=4$.
- Centroid optimization grows slowly on the tested grid.
- These are checked upper-bound witnesses, not certified minima.

---

# Experiments (2): LIMIT and LIMIT-small

Random additive single-vector embeddings cross the reported Promptriever line.

![height:300px](../../paper/figures/limit_retrieval_limit.pdf)
![height:300px](../../paper/figures/limit_retrieval_limit_small.pdf)

At $d=4096$, vanilla reaches Recall@2 $0.7060$ on LIMIT and $0.9545$ on LIMIT-small without using labels.

The packaged LIMIT top-$2$ tasks can also be exactly overfit in $\mathbb{R}^4$ by the cyclic-polytope construction.

---

# Conclusion

**Exact MED:** cyclic-polytopal neighborliness gives explicit $\Theta(k)$-dimensional witnesses for arbitrary top-$k$ answer sets.

**Robust MED:** normalized margins change the regime: feasible gaps are capped by $\epsilon_\star(m,k)$, and Gaussian centroids give $O(k^2\log m)$ dimensions at the $c/\sqrt{k}$ scale.

**Empirical interpretation:** synthetic and LIMIT failures are not exact-capacity failures.

The remaining difficulties lie in learning, tokenization, objectives, conditioning, finite precision, and optimization.
