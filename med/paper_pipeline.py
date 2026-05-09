from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from med.cyclic_polytope.experiment import Experiment as CyclicPolytopeExperiment
from med.defaults import DEFAULT_M_VALUES
from med.mean_embedding.experiment import Experiment as MeanEmbeddingExperiment
from med.plotting import invert_wbnl_curve, set_paper_style, wbnl_critical_m

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K = 2
INNER_PRODUCT = "inner_product"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "upper_bound_witness"
DEFAULT_PAPER_TABLE_DIR = REPO_ROOT / "paper" / "table"
DEFAULT_PAPER_FIGURE_DIR = REPO_ROOT / "paper" / "figure"


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


def _entry_for_dimension(search_path: list[dict[str, Any]], dimension: int) -> dict[str, Any]:
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
        cyclic = cyclic_by_m[m]
        mean = mean_by_m[m]
        cyclic_d = int(cyclic["med"])
        mean_d = int(mean["med"])
        cyclic_entry = _entry_for_dimension(cyclic["search_path"], cyclic_d)
        mean_entry = _entry_for_dimension(mean["search_path"], mean_d)
        total_queries = math.comb(m, k)

        summary.append(
            {
                "m": m,
                "k": k,
                "top_k_queries": total_queries,
                "cyclic_polytope_d": cyclic_d,
                "cyclic_queries_checked": cyclic_entry.get("checks", 0),
                "cyclic_total_queries": cyclic_entry.get("total_queries", total_queries),
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


def _observed_m_star(rows: list[dict[str, Any]], d_key: str) -> tuple[list[int], list[float]]:
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

    cyclic_ds, cyclic_ms = _observed_m_star(rows, "cyclic_polytope_d")
    mean_ds, mean_ms = _observed_m_star(rows, "mean_embedding_d")
    max_d = max(cyclic_ds + mean_ds + [10])
    d_range = np.linspace(1, max_d, 300)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        d_range,
        wbnl_critical_m(d_range),
        "k--",
        linewidth=2,
        label="WBNL (2025) fitted",
    )
    if cyclic_ds:
        ax1.plot(
            cyclic_ds,
            cyclic_ms,
            "s-",
            linewidth=2,
            markersize=6,
            label="Cyclic polytope witness",
        )
    if mean_ds:
        ax1.plot(
            mean_ds,
            mean_ms,
            "o-",
            linewidth=2,
            markersize=6,
            label="Centroid GD witness",
        )
    ax1.set_xlabel("Dimension $d$")
    ax1.set_ylabel("Observed supported objects $m^*$")
    ax1.set_title("Upper-bound witnesses: objects vs dimension ($k=2$)")
    ax1.set_ylim(bottom=0, top=max_m * 1.08)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig1_path = output_dir / "compare_plot1.pdf"
    fig1.savefig(fig1_path)
    plt.close(fig1)

    cyclic_m, cyclic_d = _valid_dimension_pairs(rows, "cyclic_polytope_d")
    mean_m, mean_d = _valid_dimension_pairs(rows, "mean_embedding_d")
    m_range = np.logspace(np.log10(min(5, max_m)), np.log10(max_m * 1.5), 300)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        m_range,
        invert_wbnl_curve(m_range),
        "k--",
        linewidth=2,
        label="WBNL (2025) fitted",
    )
    if cyclic_m:
        ax2.plot(
            cyclic_m,
            cyclic_d,
            "s-",
            linewidth=2,
            markersize=6,
            label="Cyclic polytope witness",
        )
    if mean_m:
        ax2.plot(
            mean_m,
            mean_d,
            "o-",
            linewidth=2,
            markersize=6,
            label="Centroid GD witness",
        )
    ax2.set_xlabel("Number of objects $m$")
    ax2.set_ylabel("Witness dimension $d$")
    ax2.set_xscale("log", base=2)
    ax2.set_title("Upper-bound witness dimensions ($k=2$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig2_path = output_dir / "compare_plot2.pdf"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    if paper_figure_dir is not None:
        paper_figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig1_path, paper_figure_dir / fig1_path.name)
        shutil.copy2(fig2_path, paper_figure_dir / fig2_path.name)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.output_root
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "k": args.k,
        "m_values": args.m_values,
        "scoring_function": INNER_PRODUCT,
        "mean_embedding": {
            "trainer": "gd",
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
        },
        "env": _env_info(),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "config.json", config)

    print(f"[PIPELINE] output_dir={run_dir}")
    print("[PIPELINE] Running cyclic-polytope witness scan")
    cyclic = CyclicPolytopeExperiment()
    cyclic.find_minimal_dimension(k_values=[args.k], m_values=args.m_values)
    cyclic_rows = _experiment_rows("cyclic_polytope", cyclic, args.k, args.m_values)

    print("[PIPELINE] Running centroid mean-embedding GD witness scan")
    mean = MeanEmbeddingExperiment(
        scoring_function=INNER_PRODUCT,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    mean.find_minimal_dimension(k_values=[args.k], m_values=args.m_values)
    mean_rows = _experiment_rows("mean_embedding", mean, args.k, args.m_values)

    summary_rows = _build_summary_rows(args.m_values, args.k, cyclic_rows, mean_rows)
    payload = {
        "pipeline": "upper_bound_witness_grid",
        "k": args.k,
        "m_values": args.m_values,
        "scoring_function": INNER_PRODUCT,
        "results": {
            "cyclic_polytope": cyclic_rows,
            "mean_embedding": mean_rows,
        },
        "summary_rows": summary_rows,
        "config": config,
    }

    _write_json(run_dir / "results.json", payload)
    _write_table_csv(run_dir / "upper_bound_witness_table.csv", summary_rows)
    _write_table_tex(run_dir / "upper_bound_witness_table.tex", summary_rows)
    return payload | {"_run_dir": str(run_dir)}


def load_payload(results_file: Path) -> dict[str, Any]:
    with results_file.open() as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and plot the k=2 upper-bound witness grid for cyclic polytope "
            "and centroid mean-embedding GD."
        )
    )
    parser.add_argument("--mode", choices=["run", "plot"], default="run")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--m_values", type=int, nargs="*", default=DEFAULT_M_VALUES)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2.0)
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
        help="Directory receiving final LaTeX tables.",
    )
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        default=DEFAULT_PAPER_FIGURE_DIR,
        help="Directory receiving final PDF figures.",
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
    if args.k != 2:
        print(f"[WARN] Paper figures are captioned for k=2; got k={args.k}.")

    if args.mode == "run":
        payload = run_pipeline(args)
        run_dir = Path(payload["_run_dir"])
    else:
        if args.results_file is None:
            raise SystemExit("--results-file is required in --mode plot")
        payload = load_payload(args.results_file)
        run_dir = args.results_file.parent

    paper_table_dir, paper_figure_dir = _paper_dirs(args)
    generate_figures(payload, run_dir, paper_figure_dir)
    _write_table_csv(run_dir / "upper_bound_witness_table.csv", payload["summary_rows"])
    _write_table_tex(run_dir / "upper_bound_witness_table.tex", payload["summary_rows"])
    if paper_table_dir is not None:
        paper_table_dir.mkdir(parents=True, exist_ok=True)
        _write_table_csv(paper_table_dir / "upper_bound_witness_table.csv", payload["summary_rows"])
        _write_table_tex(paper_table_dir / "upper_bound_witness_table.tex", payload["summary_rows"])

    print(f"[PIPELINE] Wrote results to {run_dir / 'results.json'}")
    print(f"[PIPELINE] Wrote table to {run_dir / 'upper_bound_witness_table.csv'}")
    print(f"[PIPELINE] Wrote figures to {run_dir / 'compare_plot1.pdf'} and compare_plot2.pdf")
    if paper_table_dir is not None or paper_figure_dir is not None:
        print(
            "[PIPELINE] Wrote paper artifacts to "
            f"{paper_table_dir or '<disabled>'} and {paper_figure_dir or '<disabled>'}"
        )


if __name__ == "__main__":
    main()
