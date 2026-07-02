# Finalization Log

## Release Goal

Prepare this repository for the final ICML open-source release, including:

- code for the MED experiments and LIMIT retrieval experiments;
- cached and reproducible experiment results;
- camera-ready LaTeX source and final PDF;
- 5-minute presentation slides;
- poster.

## Current Snapshot

Date: 2026-05-31

- Worktree: clean at the start of finalization (`git status --short` produced no entries).
- Code: core packages live in `med/` and `unlimit/`; tests live in `tests/`; `pyproject.toml`, `requirements.txt`, and `uv.lock` are present.
- Experiment results: canonical cached/provenance outputs remain under `results/upper_bound_witness/`, `results/unlimit/random_embeddings/`, and `results/unlimit/cyclic_overfit/`. The paper consumes exported copies from `paper/figures/` and `paper/tables/`.
- Upper-bound witness rerun: the post-margin `train_gd` rerun was interrupted before the final `m=640` row completed, so refreshed GD results should not be treated as finalized until rerun and audit complete.
- Paper source: `paper/draft_main.tex` is the canonical LaTeX entry point with section files inlined. Used support files remain under `paper/`, including styles, bibliography, figures, and the table `.tex` files consumed by the entry point. Intermediate `paper/stable/`, `paper/revisions/`, and `paper/overleaf_final/` trees have been removed.
- Paper PDF: canonical final PDF is `paper/draft_main.pdf`, built from `paper/draft_main.tex` with fixed `SOURCE_DATE_EPOCH=1780164721` and `FORCE_SOURCE_DATE=1` for reproducible comparison against the retained zip.
- Paper figures/tables: the paper source consumes release-facing copies under `paper/figures/` and `paper/tables/`; `results/` is kept as experiment provenance/cache, not as a LaTeX build dependency.
- Slides/poster: a 5-minute slide source, Beamer source/PDF, and word-by-word narration draft now exist under `presentation/icml2026_5min/`; PPTX export is not needed; poster is still pending.
- Ignore policy: `.gitignore` ignores `*.pdf` globally except release exceptions. Paper and presentation PDF exceptions are present; poster exception is still pending after the poster path is selected.

## Release Decisions

- Canonical paper source: `paper/draft_main.tex` plus support files in `paper/`.
- Canonical paper PDF path: `paper/draft_main.pdf`; build with `SOURCE_DATE_EPOCH=1780164721 FORCE_SOURCE_DATE=1`.
- Overleaf upload snapshot policy: keep `paper/overleaf_final.zip` unchanged as the final Overleaf upload snapshot and as the reference for equivalent file-manipulation sanity checks.
- Slides location and source format: `presentation/icml2026_5min/` with Markdown outline, Beamer source/PDF, and narration draft. PPTX export is intentionally omitted.
- Poster location and source format: pending.
- Experiment-result policy: pending rerun/audit decision for `train_gd` upper-bound witness results.
- Large/local artifact policy: remove intermediate drafts, deltas, and non-final PDFs from the public release after final artifacts are selected.
- Harness policy: removed redundant camera-ready harness scaffolding from the public release; README now keeps direct reproducibility commands.
- Repo-agent policy: remove tool-specific Claude instructions; keep release guidance in README/finalization notes instead.

## Checklist

- [x] Create finalization tracker.
- [x] Inventory current release-facing artifacts.
- [x] Decide canonical camera-ready paper source.
- [x] Decide concrete canonical camera-ready PDF filename/path.
- [x] Decide treatment of `paper/overleaf_final/` and `paper/overleaf_final.zip`.
- [x] Add or generate 5-minute ICML presentation slides.
- [x] Export 5-minute ICML presentation slides to PDF.
- [x] Decide PPTX export is not needed.
- [ ] Add or generate ICML poster.
- [ ] Normalize `.gitignore` exceptions for final PDFs, slides, and poster.
- [ ] Complete or explicitly freeze the `train_gd` upper-bound witness results.
- [ ] Regenerate paper-facing tables/figures from finalized cached results.
- [x] Flatten camera-ready paper source into `paper/` and inline section files.
- [x] Verify flattened paper PDF against the retained Overleaf zip reference.
- [x] Run the test suite.
- [ ] Run direct paper/result verification checks after harness removal.
- [ ] Audit README for final public-release instructions.
- [x] Remove redundant harness code, manifests, and docs after final direct reproduction commands are documented.
- [ ] Audit packaging/dependencies for public install and reproducibility.
- [ ] Audit license, citation, and artifact provenance notes.
- [x] Remove redundant non-final PDF artifacts.
- [x] Remove or quarantine intermediate drafts, deltas, and other non-release local artifacts.
- [x] Remove Claude-specific repo instruction file.
- [ ] Final git status and tracked-file review.

## Progress Log

- 2026-05-31: Started final open-source release pass. Initial inventory found tracked code, tests, result artifacts, Overleaf final source/zip, and tracked `results/latex/icml2026.pdf`; no tracked slides or poster artifacts were found.
- 2026-05-31: Decided that intermediate drafts, deltas, and non-final PDFs should not be part of the public release. No cleanup has been executed yet.
- 2026-05-31: Decided that redundant camera-ready harness scaffolding should also be removed from the public release, after direct reproduction commands are kept in the README or scripts.
- 2026-05-31: Removed redundant draft/result PDFs. Retained `paper/icml2026.pdf` until the canonical final PDF path is decided, retained `paper/overleaf_final/figures/*.pdf` because the Overleaf source references them, and retained paper-facing result figures under `results/**`.
- 2026-05-31: Removed redundant harness scaffold: `harness/`, `med/harness.py`, `tests/test_harness.py`, README harness section, and the `med-harness` console script.
- 2026-05-31: Added 5-minute presentation draft files under `presentation/icml2026_5min/`: `slides.md` and `narration.md`.
- 2026-05-31: Ran `uv run pytest tests -q`; result: 64 passed.
- 2026-05-31: Initially decided that the canonical final paper source was `paper/overleaf_final/`, with `paper/overleaf_final.zip` as the upload snapshot. This was later superseded by flattening the source into `paper/` while keeping the zip as the reference snapshot.
- 2026-05-31: Added Beamer source `presentation/icml2026_5min/slides_beamer.tex` and exported `presentation/icml2026_5min/icml2026_5min_beamer.pdf`. Updated `.gitignore` to keep Beamer auxiliary files ignored and allow presentation PDF/PPTX release artifacts.
- 2026-05-31: Decided that PPTX export is not needed for the 5-minute presentation; the Beamer PDF is the presentation export artifact.
- 2026-05-31: Removed `paper/stable/` and `paper/revisions/`. Updated README paper layout/build instructions and presentation source notes to point to `paper/overleaf_final/`; this was later superseded by the flattened `paper/` layout.
- 2026-05-31: Removed `CLAUDE.md` as part of moving the repo away from Claude-specific scaffolding.
- 2026-05-31: Moved the canonical Overleaf source files from `paper/overleaf_final/` into `paper/`, inlined section inputs into `paper/draft_main.tex`, removed the redundant section tree, and kept only the figure PDFs referenced by the flattened source.
- 2026-05-31: Kept `paper/overleaf_final.zip` unchanged as the final Overleaf upload snapshot and sanity-check reference.
- 2026-05-31: Restored the used table `.tex` inputs under `paper/tables/` because direct table inlining changed appendix table spacing. Unused table files remain removed.
- 2026-05-31: Rebuilt `paper/draft_main.pdf` from the flattened source and rebuilt `draft_main.pdf` from the retained zip in `/private/tmp`. Text extraction matched, rendered page images matched at 144 dpi, and the LaTeX log has no unresolved reference/citation warnings.
- 2026-05-31: Rebuilt canonical `paper/draft_main.pdf` with fixed `SOURCE_DATE_EPOCH=1780164721` and `FORCE_SOURCE_DATE=1`, using the retained zip mtime as the reproducibility timestamp. The resulting `paper/draft_main.pdf` is byte-identical to the PDF built from the unzipped source under the same environment: 22 pages, 428725 bytes, SHA-256 `c61ef10a838c33991a834d984e74cae149ed19b8e51d5d7257ffd1ba126916ba`.
- 2026-05-31: Moved the remaining inline tabular data from `paper/draft_main.tex` into `paper/tables/limit_promptriever_crossing.tex`. Rebuilt with the fixed reproducibility timestamp; `paper/draft_main.pdf` remains byte-identical to the zip-built reference with SHA-256 `c61ef10a838c33991a834d984e74cae149ed19b8e51d5d7257ffd1ba126916ba`.
- 2026-05-31: Cleaned stale release artifacts from `output/`, root `results.json`, old `results/latex/`, old `results/revisions/`, obsolete timestamped `results/cyclic_polytope/`, obsolete `results/gd/`, and stale `results/mean_embedding/`. Kept canonical result provenance under `results/upper_bound_witness/`, `results/unlimit/random_embeddings/`, and `results/unlimit/cyclic_overfit/`.
- 2026-06-01: Normalized the 5-minute presentation release surface to one canonical tracked PDF, `presentation/icml2026_5min/icml2026_5min_beamer.pdf`, with `slides_beamer.pdf` treated as an ignored LaTeX build output. Added README instructions for rebuilding the Beamer deck and copying the canonical release PDF.
- 2026-06-01: Reworked the 5-minute Beamer deck and narration to the 7-page outline: settings with MED/RMED illustration, MED construction plus VC lower bound, RMED feasibility plus Gaussian upper bound, two experiment slides, and conclusion.
