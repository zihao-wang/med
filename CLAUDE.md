# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ICML 2026 paper: "$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval." Studies the minimal embeddable dimension (MED) required for perfect top-$k$ subset retrieval under inner product, cosine similarity, and Euclidean distance scoring functions. Proves $d = \Theta(k)$ tight bounds independent of universe size $m$, reframing retrieval limits from approximability to learnability.

## Build

```bash
cd paper && latexmk -pdf icml2026.tex
```

Or for a single pass (e.g., to resolve references):

```bash
cd paper && pdflatex icml2026.tex && bibtex icml2026 && pdflatex icml2026.tex && pdflatex icml2026.tex
```

## Paper structure

- `paper/icml2026.tex` — main entry point. Sets up document class, authors, abstract, includes `content.tex` and `appendix.tex`.
- `paper/content.tex` — the main body (introduction, problem definition, theoretical bounds, simulations, related work, conclusion).
- `paper/appendix.tex` — supplementary proofs and experimental details.
- `paper/custom_command.tex` — custom math macros (`\real`, `\med`, `\fln`, `\fcos`, `\flp`, `\medc`, etc.) and theorem environments. Also configures Python code listings styling.
- `paper/preprint.bib` — bibliography.
- `paper/icml2026.sty`, `paper/icml2026.bst` — ICML 2026 style files (provided by conference).
- `paper/math_commands.tex` — standard math command definitions (provided by ICML template).

## LaTeX macros to know

| Macro | Meaning |
|-------|---------|
| `\real` | $\mathbb{R}$ |
| `\med(m,k;s)` | Minimal Embeddable Dimension given size $m$, subset size $k$, scoring function $s$ |
| `\medc(m,k;s)` | MED in centroid setting |
| `\fln` / `\fcos` / `\flp` | Functional classes: linear / cosine / $\ell_p$ scoring |
| `\draft{text}` | Blue draft annotation |
| `\hh{text}` / `\yhc{text}` | Red/green author comments |

## Code (future)

The `code/` directory will be added for numerical simulations (MED bound verification, centroid embedding experiments). Python listing style is already configured in `custom_command.tex`.
