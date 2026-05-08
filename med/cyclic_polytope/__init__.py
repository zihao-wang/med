"""
cyclic_polytope: Moment-curve construction for MED lower bounds.

Modules:
- checker: CyclicPolytopeChecker wrapping verify_construction as a FeasibilityChecker
- cli: package CLI for minimal-dimension sweeps over m and k
- construct: polynomial-based query construction for perfect top-k retrieval
- experiment: experiment orchestration (delegates to shared Experiment)
- generator: create cyclic polytope vertex sets via moment curve
"""
