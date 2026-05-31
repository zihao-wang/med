# Five-minute narration draft

Read the quoted paragraphs word for word. The timings are targets for a roughly five-minute video and assume a calm conference-recording pace.

## Slide 1 - Title, 0:00-0:30

"This talk is about Minimal Embeddable Dimension, or MED. The question is simple: if a retrieval system stores objects as vectors, how many dimensions are theoretically needed before every top-k answer set can be represented? The main result is that exact separability is much cheaper than one might expect. For answer sets of size at most k, two k dimensions are enough for inner-product retrieval."

## Slide 2 - Problem, 0:30-1:08

"Here is the formal version of the question. We have a universe of m objects. Each object is embedded as a vector in R to the d. For every subset S with size between one and k, we want a query vector and a threshold so that exactly the objects in S score above the threshold. If the answer size is known, this is the same as returning that many highest-scoring objects. MED is the smallest dimension where such a configuration exists."

## Slide 3 - Exact MED result, 1:08-1:52

"The exact MED answer is Theta of k for the standard scoring rules. For inner product, the lower bound is k minus one and the upper bound is two k. Euclidean distance has the same bounds. Cosine similarity costs at most one additional dimension, giving an upper bound of two k plus one. The important point is what is not in this table: there is no dependence on m, up to constants, for exact threshold retrieval."

## Slide 4 - Construction, 1:52-2:36

"The upper bound is constructive. Put the m objects on the moment curve in R to the two k, using coordinates t, t squared, up through t to the two k. Now fix any desired answer set S. Build the polynomial whose roots are exactly the t values for the selected objects, then square that polynomial. The squared polynomial is zero on S and positive everywhere else. Using its coefficients as a query vector makes the selected objects share the maximum score, while every unselected object scores lower."

## Slide 5 - Robust margins, 2:36-3:32

"This does not mean practical retrieval is solved. Exact separability can use extremely small gaps, and those gaps may be numerically fragile. To study that issue, the paper defines robust MED. Now all object and query vectors are normalized, and selected objects must beat unselected objects by a fixed score gap epsilon. In this regime, m comes back through a packing lower bound. There is also a feasibility ceiling: for large m, the margin cannot be larger than order one over square root k. So robustness is a genuinely different problem from exact embeddability."

## Slide 6 - Evidence, 3:32-4:32

"The experiments are best read as upper-bound witnesses, not as certified minima. For k equals two, the exact construction predicts dimension four, and the cyclic-polytope checker verifies all two hundred four thousand four hundred eighty pair queries at m equals six hundred forty. The centroid gradient-descent witness is more constrained; in the cached paper table it reaches zero violations at dimension twenty four for m equals six hundred forty. The LIMIT retrieval runs show a separate practical lesson. At dimension four thousand ninety six, handmade random token sums reach Recall at two of point nine nine eight, vanilla word tokens reach point seven zero six, and Qwen token ids reach point two six seven five. Geometry helps, but tokenization and construction still strongly shape realized retrieval."

## Slide 7 - Takeaway, 4:32-5:00

"The takeaway is the separation. For exact threshold retrieval, ambient dimension alone is not the bottleneck: dimension proportional to k is enough, with explicit query vectors. For robust and learned retrieval, the hard parts are margins, conditioning, finite precision, tokenization, and learning the query map. The repository packages the constructions, experiments, LIMIT runs, and paper-facing artifacts needed to reproduce that distinction."
