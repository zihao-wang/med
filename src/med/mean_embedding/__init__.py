"""
mean_embedding: Subset-as-mean embedding framework for minimal dimension search.

Modules:
- checker: MeanEmbeddingChecker wrapping full-batch GD as a FeasibilityChecker
- experiment: experiment orchestration (delegates to shared Experiment)
- trainer_gd: full-batch GD trainer and loss computation
"""
