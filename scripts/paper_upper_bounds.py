from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from med.cyclic_polytope.checker import CyclicPolytopeChecker
from med.defaults import DEFAULT_M_VALUES
from med.mean_embedding.checker import MeanEmbeddingChecker
from med.plotting import invert_wbnl_curve, set_paper_style
from med.search import binary_search_med

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K = 2
INNER_PRODUCT = "inner_product"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "upper_bound_witness"
DEFAULT_PAPER_TABLE_DIR = None
DEFAULT_PAPER_FIGURE_DIR = None
SUMMARY_CSV_NAME = "upper_bound_witness_table.csv"
RESULTS_JSON_NAME = "results.json"
PAPER_DIMENSION_FIT_NAME = "top2_dimension_fit.pdf"

SUMMARY_INT_FIELDS = {
    "m",
    "k",
    "top_k_queries",
    "cyclic_polytope_d",
    "cyclic_queries_checked",
    "cyclic_total_queries",
    "mean_embedding_d",
    "mean_embedding_violations",
}
SUMMARY_FLOAT_FIELDS = {
    "cyclic_checked_fraction",
    "cyclic_time",
    "mean_embedding_time",
}


def _env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def _experiment_rows(
    experiment_name: str,
    experiment: Any,
    k: int,
    m_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in m_values:
        rows.append(
            {
                "experiment": experiment_name,
                "m": m,
                "k": k,
                "med": experiment.minimal_dimensions.get(k, {}).get(m, -1),
                "search_path": experiment.search_paths.get(k, {}).get(m, []),
                "time": experiment.timings.get(k, {}).get(m, -1),
            }
        )
    return rows


def _entry_for_dimension(
    search_path: list[dict[str, Any]], dimension: int
) -> dict[str, Any]:
    for entry in search_path:
        if entry.get("dimension") == dimension:
            return entry
    return {}


def _build_summary_rows(
    m_values: list[int],
    k: int,
    cyclic_rows: list[dict[str, Any]],
    mean_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cyclic_by_m = {row["m"]: row for row in cyclic_rows}
    mean_by_m = {row["m"]: row for row in mean_rows}
    summary: list[dict[str, Any]] = []

    for m in m_values:
        if m not in cyclic_by_m or m not in mean_by_m:
            continue
        cyclic = cyclic_by_m[m]
        mean = mean_by_m[m]
        cyclic_d = int(cyclic["med"])
        mean_d = int(mean["med"])
        if cyclic_d <= 0 or mean_d <= 0:
            continue
        cyclic_entry = _entry_for_dimension(cyclic["search_path"], cyclic_d)
        mean_entry = _entry_for_dimension(mean["search_path"], mean_d)
        total_queries = sum(math.comb(m, subset_size) for subset_size in range(1, k + 1))

        summary.append(
            {
                "m": m,
                "k": k,
                "top_k_queries": total_queries,
                "cyclic_polytope_d": cyclic_d,
                "cyclic_queries_checked": cyclic_entry.get("checks", 0),
                "cyclic_total_queries": cyclic_entry.get(
                    "total_queries", total_queries
                ),
                "cyclic_checked_fraction": cyclic_entry.get("checked_fraction", 0.0),
                "cyclic_time": cyclic["time"],
                "mean_embedding_d": mean_d,
                "mean_embedding_violations": mean_entry.get("violations"),
                "mean_embedding_time": mean["time"],
            }
        )

    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "m",
        "k",
        "top_k_queries",
        "cyclic_polytope_d",
        "cyclic_queries_checked",
        "cyclic_total_queries",
        "cyclic_checked_fraction",
        "cyclic_time",
        "mean_embedding_d",
        "mean_embedding_violations",
        "mean_embedding_time",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_cell(value: str | None, field: str) -> Any:
    if value in (None, "", "--"):
        return None
    if field in SUMMARY_INT_FIELDS:
        return int(value)
    if field in SUMMARY_FLOAT_FIELDS:
        return float(value)
    return value


def _load_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = [
            {field: _parse_csv_cell(value, field) for field, value in row.items()}
            for row in csv.DictReader(f)
        ]

    if not rows:
        raise ValueError(f"No rows found in summary CSV: {path}")
    return rows


def _payload_from_summary_csv(path: Path) -> dict[str, Any]:
    summary_rows = _load_summary_csv(path)
    m_values = [int(row["m"]) for row in summary_rows]
    k_values = {int(row["k"]) for row in summary_rows if row.get("k") is not None}
    if len(k_values) > 1:
        raise ValueError(f"Expected one k value in summary CSV, found {sorted(k_values)}")
    k = next(iter(k_values), DEFAULT_K)

    return {
        "pipeline": "upper_bound_witness_grid",
        "k": k,
        "m_values": m_values,
        "scoring_function": INNER_PRODUCT,
        "results": {},
        "summary_rows": summary_rows,
        "config": {
            "resume_source": str(path),
            "resume_format": "summary_csv",
        },
    }


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "$m$ & $\\binom{m}{k}$ & Cyclic $d$ & Cyclic checked & "
        "MED-C $d$ & MED-C violations & MED-C time (s) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['m']} & "
            f"{row['top_k_queries']} & "
            f"{row['cyclic_polytope_d']} & "
            f"{row['cyclic_queries_checked']}/{row['cyclic_total_queries']} & "
            f"{row['mean_embedding_d']} & "
            f"{row['mean_embedding_violations']} & "
            f"{_format_float(row['mean_embedding_time'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _write_payload_artifacts(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    paper_table_dir: Path | None = None,
) -> None:
    cyclic_rows = payload["results"].get("cyclic_polytope", [])
    mean_rows = payload["results"].get("mean_embedding", [])
    payload["summary_rows"] = _build_summary_rows(
        payload["m_values"],
        payload["k"],
        cyclic_rows,
        mean_rows,
    )
    _write_json(run_dir / RESULTS_JSON_NAME, payload)
    _write_table_csv(run_dir / SUMMARY_CSV_NAME, payload["summary_rows"])
    _write_table_tex(run_dir / "upper_bound_witness_table.tex", payload["summary_rows"])
    if paper_table_dir is not None:
        paper_table_dir.mkdir(parents=True, exist_ok=True)
        _write_table_csv(
            paper_table_dir / "upper_bound_witness_table.csv",
            payload["summary_rows"],
        )
        _write_table_tex(
            paper_table_dir / "upper_bound_witness_table.tex",
            payload["summary_rows"],
        )


def _upsert_result_row(
    payload: dict[str, Any],
    experiment_name: str,
    row: dict[str, Any],
) -> None:
    rows = payload["results"].setdefault(experiment_name, [])
    for index, existing in enumerate(rows):
        if existing["m"] == row["m"] and existing["k"] == row["k"]:
            rows[index] = row
            return
    rows.append(row)


def _run_checker_sweep(
    *,
    experiment_name: str,
    checker_factory: Any,
    payload: dict[str, Any],
    run_dir: Path,
    paper_table_dir: Path | None,
    k: int,
    m_values: list[int],
    left0: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_minimal = left0
    for m in m_values:
        t0 = time.perf_counter()
        print("#" * 10 + " new task " + "#" * 10)
        print(f"[EXP] Finding minimal dimension for m={m}, k={k}")
        print("#" * 30)

        checker = checker_factory(m, k)

        def on_check(_entry: dict[str, Any], search_path: list[dict[str, Any]]) -> None:
            partial_row = {
                "experiment": experiment_name,
                "m": m,
                "k": k,
                "med": -1,
                "search_path": list(search_path),
                "time": round(time.perf_counter() - t0, 1),
                "status": "running",
            }
            _upsert_result_row(payload, experiment_name, partial_row)
            _write_payload_artifacts(
                run_dir,
                payload,
                paper_table_dir=paper_table_dir,
            )

        result = binary_search_med(
            checker,
            left0=last_minimal,
            verbose=True,
            on_check=on_check,
        )
        med = result.get("med")
        row = {
            "experiment": experiment_name,
            "m": m,
            "k": k,
            "med": med if med is not None else -1,
            "search_path": result["search_path"],
            "time": result.get(
                "total_time",
                round(time.perf_counter() - t0, 1),
            ),
            "status": "complete",
        }
        _upsert_result_row(payload, experiment_name, row)
        rows = list(payload["results"][experiment_name])
        if med is not None and med > 0:
            last_minimal = max(0, med - 1)
        _write_payload_artifacts(
            run_dir,
            payload,
            paper_table_dir=paper_table_dir,
        )

        print(f"[RESULT] minimal dimension @ k={k} & m={m} is {med}")
        print("#" * 30)
    return rows


def _observed_m_star(
    rows: list[dict[str, Any]], d_key: str
) -> tuple[list[int], list[float]]:
    pairs = [
        (int(row[d_key]), int(row["m"]))
        for row in rows
        if isinstance(row.get(d_key), int) and int(row[d_key]) > 0
    ]
    if not pairs:
        return [], []

    d_values = list(range(1, max(d for d, _ in pairs) + 1))
    m_values: list[float] = []
    for d in d_values:
        supported = [m for d_star, m in pairs if d_star <= d]
        m_values.append(float(max(supported)) if supported else np.nan)
    return d_values, m_values


def _valid_dimension_pairs(
    rows: list[dict[str, Any]], d_key: str
) -> tuple[list[int], list[int]]:
    pairs = [
        (int(row["m"]), int(row[d_key]))
        for row in rows
        if isinstance(row.get(d_key), int) and int(row[d_key]) > 0
    ]
    pairs.sort()
    return [m for m, _ in pairs], [d for _, d in pairs]


def generate_figures(
    payload: dict[str, Any],
    output_dir: Path,
    paper_figure_dir: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = payload["summary_rows"]
    max_m = max(row["m"] for row in rows)

    cyclic_m, cyclic_d = _valid_dimension_pairs(rows, "cyclic_polytope_d")
    mean_m, mean_d = _valid_dimension_pairs(rows, "mean_embedding_d")
    m_range = np.logspace(np.log10(min(5, max_m)), np.log10(max_m * 1.5), 300)

    fig2, ax2 = plt.subplots(figsize=(3.25, 2.25))
    ax2.plot(
        m_range,
        invert_wbnl_curve(m_range),
        "k--",
        linewidth=1.2,
        label="WBNL (2025) fitted",
    )
    if cyclic_m:
        ax2.plot(
            cyclic_m,
            cyclic_d,
            "s-",
            linewidth=1.2,
            markersize=3.5,
            label="Cyclic polytope witness",
        )
    if mean_m:
        ax2.plot(
            mean_m,
            mean_d,
            "o-",
            linewidth=1.2,
            markersize=3.5,
            label="Centroid GD witness",
        )
    if len(mean_m) >= 2:
        fit_coef = np.polyfit(np.log2(np.array(mean_m, dtype=float)), np.array(mean_d, dtype=float), 1)
        fit_d = np.polyval(fit_coef, np.log2(m_range))
        ax2.plot(
            m_range,
            fit_d,
            color="tab:blue",
            linestyle=":",
            linewidth=1.1,
            label="Centroid log fit",
        )
    ax2.set_xlabel("Number of objects $m$")
    ax2.set_ylabel("Witness dimension $d$")
    ax2.set_xscale("log", base=2)
    m_ticks = sorted(set(cyclic_m + mean_m))
    ax2.set_xticks(m_ticks)
    ax2.set_xticklabels([str(tick) for tick in m_ticks])
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=7, handlelength=1.4)

    fig2.tight_layout(pad=0.2)
    dimension_fit_path = output_dir / PAPER_DIMENSION_FIT_NAME
    fig2.savefig(dimension_fit_path)
    plt.close(fig2)

    if paper_figure_dir is not None:
        paper_figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dimension_fit_path, paper_figure_dir / dimension_fit_path.name)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.output_root
    run_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir, _paper_figure_dir = _paper_dirs(args)

    config = {
        "k": args.k,
        "m_values": args.m_values,
        "scoring_function": INNER_PRODUCT,
        "mean_embedding": {
            "trainer": "gd",
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "margin": args.margin,
            "constraint_chunk_size": args.constraint_chunk_size,
            "show_progress": False,
        },
        "env": _env_info(),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config)

    payload: dict[str, Any] = {
        "pipeline": "upper_bound_witness_grid",
        "k": args.k,
        "m_values": args.m_values,
        "scoring_function": INNER_PRODUCT,
        "results": {
            "cyclic_polytope": [],
            "mean_embedding": [],
        },
        "summary_rows": [],
        "config": config,
    }
    _write_payload_artifacts(run_dir, payload, paper_table_dir=paper_table_dir)

    print(f"[PIPELINE] output_dir={run_dir}")
    print("[PIPELINE] Running cyclic-polytope witness scan")
    print("#" * 10 + f" Starting k={args.k} " + "#" * 10)
    cyclic_rows = _run_checker_sweep(
        experiment_name="cyclic_polytope",
        checker_factory=lambda m, k: CyclicPolytopeChecker(m=m, k=k),
        payload=payload,
        run_dir=run_dir,
        paper_table_dir=paper_table_dir,
        k=args.k,
        m_values=args.m_values,
    )

    print("[PIPELINE] Running centroid mean-embedding GD witness scan")
    print("#" * 10 + f" Starting k={args.k} " + "#" * 10)
    mean_rows = _run_checker_sweep(
        experiment_name="mean_embedding",
        checker_factory=lambda m, k: MeanEmbeddingChecker(
            m=m,
            k=k,
            scoring_function=INNER_PRODUCT,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            margin=args.margin,
            constraint_chunk_size=args.constraint_chunk_size,
            show_progress=False,
        ),
        payload=payload,
        run_dir=run_dir,
        paper_table_dir=paper_table_dir,
        k=args.k,
        m_values=args.m_values,
    )

    payload["results"]["cyclic_polytope"] = cyclic_rows
    payload["results"]["mean_embedding"] = mean_rows
    _write_payload_artifacts(run_dir, payload, paper_table_dir=paper_table_dir)
    return payload | {"_run_dir": str(run_dir)}


def load_payload(results_file: Path) -> dict[str, Any]:
    if results_file.suffix.lower() == ".csv":
        return _payload_from_summary_csv(results_file)
    with results_file.open() as f:
        return json.load(f)


def _validate_plot_payload(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    rows = payload.get("summary_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} does not contain non-empty summary_rows")
    return payload


def _load_plot_payload(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    if args.results_file is not None:
        payload = _validate_plot_payload(
            load_payload(args.results_file), args.results_file
        )
        return payload, args.results_file.parent, args.results_file

    json_path = args.output_root / RESULTS_JSON_NAME
    csv_path = args.output_root / SUMMARY_CSV_NAME

    if json_path.exists():
        try:
            payload = _validate_plot_payload(load_payload(json_path), json_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if not csv_path.exists():
                raise SystemExit(
                    f"Could not load {json_path}: {exc}. "
                    f"No fallback CSV found at {csv_path}."
                ) from exc
            print(
                f"[PIPELINE] Could not load {json_path}; "
                f"resuming from {csv_path}"
            )
            payload = _validate_plot_payload(load_payload(csv_path), csv_path)
            return payload, csv_path.parent, csv_path
        return payload, json_path.parent, json_path

    if csv_path.exists():
        payload = _validate_plot_payload(load_payload(csv_path), csv_path)
        return payload, csv_path.parent, csv_path

    raise SystemExit(
        "--results-file was not provided and no saved results were found at "
        f"{json_path} or {csv_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and plot the k=2 upper-bound witness grid for cyclic polytope "
            "and centroid mean-embedding GD."
        )
    )
    parser.add_argument("--mode", choices=["run", "plot"], default="run")
    parser.add_argument(
        "--k",
        type=int,
        choices=[DEFAULT_K],
        default=DEFAULT_K,
        help="Paper pipeline scope is fixed to k=2.",
    )
    parser.add_argument("--m_values", type=int, nargs="*", default=DEFAULT_M_VALUES)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1e-6)
    parser.add_argument(
        "--constraint-chunk-size",
        type=int,
        default=8192,
        help="Number of subset queries per GD constraint chunk.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Deterministic directory for config.json, results.json, tables, and figures.",
    )
    parser.add_argument(
        "--paper-table-dir",
        type=Path,
        default=DEFAULT_PAPER_TABLE_DIR,
        help="Optional directory receiving copied LaTeX tables.",
    )
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        default=DEFAULT_PAPER_FIGURE_DIR,
        help="Optional directory receiving copied PDF figures.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="Deprecated parent paper directory; maps to <paper-dir>/table and <paper-dir>/figure.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "Saved results to plot from. Accepts results.json or "
            "upper_bound_witness_table.csv. Defaults to --output-root results."
        ),
    )
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="Only write figures/tables under the deterministic results directory.",
    )
    return parser.parse_args()


def _paper_dirs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if args.no_paper_copy:
        return None, None
    if args.paper_dir is not None:
        return args.paper_dir / "table", args.paper_dir / "figure"
    return args.paper_table_dir, args.paper_figure_dir


def main() -> None:
    args = parse_args()

    if args.mode == "run":
        payload = run_pipeline(args)
        run_dir = Path(payload["_run_dir"])
        results_source = run_dir / RESULTS_JSON_NAME
        wrote_results = True
    else:
        payload, run_dir, results_source = _load_plot_payload(args)
        wrote_results = False
        print(f"[PIPELINE] Resuming plot/table generation from {results_source}")

    paper_table_dir, paper_figure_dir = _paper_dirs(args)
    generate_figures(payload, run_dir, paper_figure_dir)
    _write_table_csv(run_dir / SUMMARY_CSV_NAME, payload["summary_rows"])
    _write_table_tex(run_dir / "upper_bound_witness_table.tex", payload["summary_rows"])
    if paper_table_dir is not None:
        paper_table_dir.mkdir(parents=True, exist_ok=True)
        _write_table_csv(
            paper_table_dir / "upper_bound_witness_table.csv", payload["summary_rows"]
        )
        _write_table_tex(
            paper_table_dir / "upper_bound_witness_table.tex", payload["summary_rows"]
        )

    if wrote_results:
        print(f"[PIPELINE] Wrote results to {results_source}")
    else:
        print(f"[PIPELINE] Loaded results from {results_source}")
    print(f"[PIPELINE] Wrote table to {run_dir / SUMMARY_CSV_NAME}")
    print(
        "[PIPELINE] Wrote figures to "
        f"{run_dir / PAPER_DIMENSION_FIT_NAME}"
    )
    if paper_table_dir is not None or paper_figure_dir is not None:
        print(
            "[PIPELINE] Wrote paper artifacts to "
            f"{paper_table_dir or '<disabled>'} and {paper_figure_dir or '<disabled>'}"
        )


if __name__ == "__main__":
    main()
