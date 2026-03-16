"""
Evaluation utilities for GAN experiments.

This module re-exports the evaluation and experiment-tracking helpers
implemented in src.utils_gan so the rest of the project can import from
one clean interface.
"""

from src.utils_gan import (
    extract_features,
    compute_fid,
    cache_real_features,
    evaluate_fid,
    generate_best,
    log_experiment,
    show_results,
    plot_fid_comparison,
)

__all__ = [
    "extract_features",
    "compute_fid",
    "cache_real_features",
    "evaluate_fid",
    "generate_best",
    "log_experiment",
    "show_results",
    "plot_fid_comparison",
]