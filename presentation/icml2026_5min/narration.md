# Five-minute narration draft

Read the quoted paragraphs word for word. The timings are targets for a roughly five-minute video and assume a calm conference-recording pace.

## Slide 1 - Title, 0:00-0:25

"This talk is about Minimal Embeddable Dimension, or MED, for embedding-based top-k retrieval. The question is: how many dimensions are theoretically needed before every answer set of size at most k can be represented by vector scores? The main message is that exact geometric approximability is not the obstruction."

## Slide 2 - Settings, 0:25-1:10

"Here is the setting. We store m objects as vectors, and a query retrieves by comparing scores against a threshold. Exact MED asks for a configuration where every subset S of size between one and k can be separated from its complement by some query and threshold. If the answer size is known, this is equivalent to top size-of-S retrieval. Robust MED asks for more. Now the objects and queries are normalized, and selected objects must beat unselected objects by a fixed score gap epsilon. This margin requirement is the first place where exact separability and robust retrieval diverge."

## Slide 3 - MED, 1:10-2:05

"The upper bound comes first. Put the objects on the moment curve in R to the two k. For a target set S, form the polynomial whose roots are exactly the selected parameters, then square it and take the negative coefficients as the query vector. The selected objects score zero, and every unselected object scores strictly below zero. This gives an explicit inner-product witness in two k dimensions. The lower bound comes from VC dimension. If MED works, then among any k chosen objects it realizes every subset, so the threshold class shatters k points. Since linear thresholds in d dimensions have VC dimension d plus one, we need d at least k minus one. Together with the Euclidean and cosine reductions, this gives the displayed MED bounds."

## Slide 4 - RMED, 2:05-3:00

"Robust MED changes the regime because the margin cannot be arbitrarily large. For one through k at most m over two, any robust witness must have epsilon at most epsilon star of m and k, equal to m divided by the square root of k times m minus one times m minus k. This ceiling is tight in high dimension by a regular simplex construction. In the large-universe retrieval regime, where m over k goes to infinity, the ceiling is order one over square root k. At that feasible scale, a Gaussian centroid construction gives an upper bound: sample random unit object vectors in dimension on the order of k squared log m, and use the normalized centroid of the selected vectors as the query. With positive probability, all selected objects beat all outsiders by a constant over square root k margin."

## Slide 5 - Experiments (1), 3:00-3:45

"The first experiment checks the synthetic top-two query setting. For k equals two, the exact cyclic-polytope construction predicts dimension four, independent of the number of objects, and the blue line shows that witness. The centroid gradient-descent witness is a different, more restricted protocol, but it still grows slowly on this grid and stays far below the fitted Weller baseline curve. These plotted points are upper-bound witnesses, not certified minima, so failed optimization should not be read as proof of infeasibility."

## Slide 6 - Experiments (2), 3:45-4:30

"The second experiment studies LIMIT and LIMIT-small. We use random additive single-vector embeddings: tokenize a document or query, assign each token a random vector, sum the token vectors, and rank by inner product. The dotted lines are the reported Promptriever single-vector baselines. All three tokenizations cross those lines. The vanilla tokenizer is the key control because it is label-unaware and unsupervised, yet at dimension four thousand ninety six it reaches Recall at two of point seven zero six on LIMIT and point nine five four five on LIMIT-small. The packaged top-two instances can also be exactly overfit in R to the four by the cyclic-polytope construction."

## Slide 7 - Conclusion, 4:30-5:00

"The conclusion is the separation. Exact MED is low-dimensional: cyclic-polytopal neighborliness gives explicit Theta of k witnesses for arbitrary top-k answer sets. Robust MED is different: finite-m score gaps are capped by epsilon star, and feasible large-universe margins have an order k squared log m Gaussian centroid witness. The empirical failures here are therefore not failures of exact geometric capacity. The remaining difficulties are learning, tokenization, objectives, conditioning, finite precision, and optimization."
