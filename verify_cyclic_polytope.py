from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm, trange

from src.cyclic_polytope.generator import generate_cyclic_polytope_configuration
from src.cyclic_polytope.verification import is_face_linear_feasible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify linear separability of cyclic polytope embeddings")
    parser.add_argument("--m", type=int, default=64, help="Number of points on the moment curve")
    parser.add_argument("--n", type=int, default=5, help="Ambient dimension")
    parser.add_argument("--tol", type=float, default=1e-9, help="Feasibility tolerance")
    parser.add_argument("--pairs_only", action="store_true", help="Check only pairs (i,j) as faces")
    return parser.parse_args()


def pair_iter(m: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs.append((i, j))
    return pairs


def main() -> None:
    args = parse_args()
    X = generate_cyclic_polytope_configuration(args.m, args.n)

    if args.pairs_only:
        good_prefix_exists = False
        for i in trange(args.m, desc="Checking prefix pairs"):
            satisfies_i = True
            for j in range(i + 1, args.m):
                res = is_face_linear_feasible(X, [i, j], tol=args.tol)
                if res.status == "unsat":
                    satisfies_i = False
                    break
            if satisfies_i:
                good_prefix_exists = True
        print("Pair-face prefix exists:" if good_prefix_exists else "No pair-face prefix found")
    else:
        # Simple full pair sweep
        num_sat = 0
        for i, j in tqdm(pair_iter(args.m), desc="Checking all pairs"):
            res = is_face_linear_feasible(X, [i, j], tol=args.tol)
            num_sat += int(res.status != "unsat")
        print(f"Satisfiable pairs: {num_sat}")


if __name__ == "__main__":
    main()
