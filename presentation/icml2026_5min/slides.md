---
marp: true
title: "R^{2k} is Theoretically Large Enough for Embedding-based Top-k Retrieval"
description: "Five-minute ICML 2026 video recording deck source"
paginate: true
math: mathjax
---

<!--
Target: 5-minute ICML-style video, 7--8 slides.

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

## Minimal Embeddable Dimension (MED)

Zihao Wang, Hang Yin, Lihui Liu, Hanghang Tong, Yangqiu Song, Ginny Wong, Simon See

**Main message:** exact geometric approximability is not the obstruction.

---

# The question behind vector retrieval

Embedding retrieval stores $m$ objects as vectors in $\mathbb{R}^d$.

A query should recover any answer set

$$
S \subseteq X,\qquad 1 \le |S| \le k
$$

by score comparison and a threshold.

**MED** is the smallest dimension $d$ where this is possible for every such $S$.

When $|S|$ is known, thresholding is equivalent to top-$|S|$ retrieval.

---

# Exact MED is $\Theta(k)$, not a function of $m$

| Scoring rule | Lower bound | Upper bound |
|---|---:|---:|
| Inner product | $k-1$ | $2k$ |
| Euclidean distance | $k-1$ | $2k$ |
| Cosine similarity | $k-1$ | $2k+1$ |

The lower bounds come from VC dimension.

The upper bounds are explicit constructions.

**Takeaway:** for exact threshold retrieval, the universe size $m$ does not drive the dimension up.

---

# Why $2k$ dimensions are enough

Place objects on the moment curve:

$$
v_i = (t_i, t_i^2, \ldots, t_i^{2k})
$$

For a target subset $S$, form

$$
P_S(t)=\prod_{i\in S}(t-t_i).
$$

Use the coefficients of $-P_S(t)^2$ as the query vector.

Selected objects score at the shared maximum; every other object scores lower.

This is the cyclic-polytope neighborliness witness in closed form.

---

# Margins change the problem

Exact retrieval can rely on arbitrarily small gaps.

Robust MED asks for unit vectors and a normalized gap $\epsilon$:

$$
\min_{i\in S}\langle u_S,v_i\rangle
\ge
\max_{j\notin S}\langle u_S,v_j\rangle+\epsilon .
$$

Then $m$ reappears:

$$
\operatorname{RMED}(m,k,\epsilon)
\ge
\frac{\log {m \choose k}}{\log(1+2/\epsilon)} .
$$

Finite-$m$ score gaps are capped by

$$
\epsilon_\star(m,k)=\frac{m}{\sqrt{k(m-1)(m-k)}}.
$$

Our RMED shorthand uses the large-universe regime $m/k\to\infty$, where

$$
\epsilon_\star(m,k)\sim \frac{1}{\sqrt{k}}.
$$

At the feasible $c/\sqrt{k}$ scale, Gaussian centroid witnesses give
$O(k^2\log m)$ dimensions.

---

# Evidence: witnesses, not minima

For $k=2$, the exact construction uses $d=4$.

| Paper witness grid | Final row |
|---|---|
| Cyclic polytope | exact $d=4$ witness for arbitrary top-2 answer sets |
| Centroid GD | zero violations at $d=24$ for $m=640$ |

LIMIT retrieval shows the practical side:

| LIMIT full, Recall@2 at $d=4096$ | Result |
|---|---:|
| Handmade token sums | 0.9980 |
| Vanilla word tokens | 0.7060 |
| Qwen token ids | 0.2675 |

Tokenizer geometry and construction choices still matter.

---

# What this means for retrieval

Exact threshold retrieval:

**Ambient dimension alone is not the bottleneck.**

Robust or learned retrieval:

**The bottlenecks are margin, learning, tokenization, objectives, conditioning, finite precision, and optimization.**

The repository provides:

- exact cyclic-polytope witnesses;
- centroid/Gaussian witness experiments;
- LIMIT random-token retrieval runs;
- paper-ready result tables and figures.
