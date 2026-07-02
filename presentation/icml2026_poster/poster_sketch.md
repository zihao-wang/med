# ICML Poster Sketch

This folder holds a first visual sketch for the ICML poster:

- `poster_sketch.svg`: landscape 4:3 layout sketch, sized as a scalable vector.
- `poster_sketch_preview.png`: rendered 1600x1200 preview for quick viewing.
- `poster.tex`: editable 48in x 36in LaTeX poster source.
- `icml2026_poster.pdf`: compiled poster PDF.
- `icml2026_poster_preview.png`: rendered PDF preview.
- `logos/`: downloaded logo originals and PNG conversions used by LaTeX.

Build the LaTeX poster from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error poster.tex
cp poster.pdf icml2026_poster.pdf
```

## Read Path

1. Title and one-sentence thesis.
2. Problem definition: MED asks whether all answer sets up to size `k` are retrievable by score comparison.
3. Exact theory: moment-curve / squared-polynomial construction gives `k - 1 <= MED <= 2k` for inner product and Euclidean scores, with `2k + 1` for cosine.
4. Robust theory: normalized margins introduce the finite-`m` ceiling `epsilon_star = m / sqrt(k(m-1)(m-k))`; at the feasible `1 / sqrt(k)` scale, Gaussian centroids give `O(k^2 log m)`.
5. Experiments: synthetic top-2 confirms the `d = 4` cyclic-polytope witness, and LIMIT / LIMIT-small random token-sum baselines cross the reported single-vector embedding baseline.
6. Bottom-line interpretation: failures are about learning, margins, conditioning, finite precision, and optimization, not exact geometric capacity.

## Source Assets For Final Poster

- Paper source: `paper/draft_main.tex`
- Slides source: `presentation/icml2026_5min/slides.md`
- Synthetic figure: `paper/figures/top2_dimension_fit.pdf`
- LIMIT figure: `paper/figures/limit_retrieval_limit.pdf`
- LIMIT-small figure: `paper/figures/limit_retrieval_limit_small.pdf`
- Synthetic table: `paper/tables/upper_bound_witness_table.tex`
- Random token-sum table: `paper/tables/limit_retrieval_table.tex`
- Cyclic LIMIT overfit table: `paper/tables/limit_cyclic_overfit_summary.tex`

## Next Layout Decisions

- Confirm the actual ICML poster size and whether the venue wants portrait, landscape, or virtual-first formatting.
- Decide whether the final artifact should be LaTeX, PowerPoint, or SVG/HTML exported to PDF.
- Replace the sketch charts with the three existing paper figures.
- Add a QR code to the repository or paper once the final public URL is known.
