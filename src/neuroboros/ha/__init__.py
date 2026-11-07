"""Hyperalignment: functional alignment across individuals.

Searchlight hyperalignment algorithms
    `searchlight_ridge`, `searchlight_ridge_path`, and
    `searchlight_procrustes` fit a `Transform` from a subject's data to a
    template, generating one local transform per searchlight, which are then
    combined into a single whole-brain transform.
    `searchlight_fit` does the same with a custom local model.
ROI hyperalignment algorithms
    `ridge`, `ridge_path`, and `procrustes` fit a single transform between
    two data matrices, and they are typically used to hyperalign a single
    region of interest (ROI) across subjects.
Transforms
    `Transform` holds a fitted sparse transform, applies it to new data,
    and saves and loads it.
Computing your own templates
    `compute_template` computes a template of several subjects' data for a
    set of features, typically for a brain region.
    `compute_searchlight_template` combines local templates across
    searchlights into a single template for the whole brain.

`searchlight_weights`, `ridge_grid`, and `ensemble_ridge` are older helpers
kept for existing analyses.
"""

from .procrustes import procrustes
from .ridge import ensemble_ridge, ridge, ridge_grid, ridge_path
from .searchlight import compute_searchlight_weights as searchlight_weights
from .searchlight import (
    searchlight_fit,
    searchlight_procrustes,
    searchlight_ridge,
    searchlight_ridge_path,
)
from .template import compute_template, compute_searchlight_template
from .transform import Transform

__all__ = [
    "Transform",
    "compute_template",
    "compute_searchlight_template",
    "searchlight_ridge",
    "searchlight_ridge_path",
    "searchlight_procrustes",
    "searchlight_fit",
    "ridge",
    "ridge_path",
    "procrustes",
    "searchlight_weights",
    "ridge_grid",
    "ensemble_ridge",
]
