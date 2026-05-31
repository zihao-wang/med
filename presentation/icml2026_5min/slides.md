---
marp: true
title: "R^{2k} is Theoretically Large Enough for Embedding-based Top-k Retrieval"
description: "Five-minute ICML 2026 video recording deck source"
paginate: true
math: mathjax
---

<!--
Target: 5-minute ICML-style video, 7 slides.

Primary sources:
- README.md
- paper/draft_main.tex
- results/upper_bound_witness/upper_bound_witness_table.tex
- results/unlimit/random_embeddings/limit_retrieval_table.tex
- results/unlimit/cyclic_overfit/limit_cyclic_overfit_summary.tex

Optional existing visual assets for rendering:
- ../../results/upper_bound_witness/compare_plot1.pdf
- ../../results/upper_bound_witness/compare_plot2.pdf
- ../../results/unlimit/random_embeddings/limit_retrieval_limit.pdf
- ../../results/unlimit/random_embeddings/limit_retrieval_limit_small.pdf
-->

<!-- _class: lead -->

# $\mathbb{R}^{2k}$ is theoretically large enough for embedding-based top-$k$ retrieval

## Minimal Embeddable Dimension (MED)

Zihao Wang, Hang Yin, Lihui Liu, Hanghang Tong, Yangqiu Song, Ginny Wong, Simon See

**Main message:** exact separability is cheap; robust, learnable retrieval is the hard part.

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

And margins above

$$
\epsilon_\star(m,k)=\frac{m}{\sqrt{k(m-1)(m-k)}}\sim \frac{1}{\sqrt{k}}
$$

are infeasible when $m\gg k$.

---

# Evidence: witnesses, not minima

For $k=2$, the exact construction uses $d=4$.

| Paper witness grid | Final row |
|---|---|
| Cyclic polytope | $d=4$, $204{,}480/204{,}480$ pair queries checked at $m=640$ |
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

**The bottlenecks are margin, conditioning, finite precision, tokenization, and query-map learnability.**

The repository provides:

- exact cyclic-polytope witnesses;
- centroid/Gaussian witness experiments;
- LIMIT random-token retrieval runs;
- paper-ready result tables and figures.
