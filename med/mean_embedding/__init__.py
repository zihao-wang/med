"""
mean_embedding: Subset-as-mean embedding framework for minimal dimension search.

Modules:
- checker: MeanEmbeddingChecker wrapping Trainer/SGDTrainer as a FeasibilityChecker
- cli: package CLI for custom minimal-dimension sweeps
- compare_plots: paper compare-plot experiment and plotting entry point
- experiment: experiment orchestration (delegates to shared Experiment)
- trainer_gd: full-batch GD trainer and loss computation
- trainer_sgd: scalable stochastic trainer using random k-subsets
"""
