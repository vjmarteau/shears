import logging
from typing import Any

import pandas as pd
import numpy as np
from anndata import AnnData
from shears.util._wrangling import _is_na_strict

logg = logging.getLogger(__name__)


def _prepare_bulk_obs_and_weights(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    cell_weights_key: str,
    covariate_str: str | None,
    response_cols: list[str],
    max_normalize_weights: bool = True,
    init_kwargs: dict[str, Any] | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Parse the regression formula and subset bulk data for modeling."""
    init_kwargs = init_kwargs or {}

    cov = (covariate_str or "").strip(" +")
    covariate_term = f" + {cov}" if cov else ""

    if len(response_cols) == 1:
        formula = f"{response_cols[0]} ~ cell_weight{covariate_term}"
    else:
        formula = f"cell_weight{covariate_term}"

    # Extract raw covariate names from formulaic formula strings
    covariate_list = [
        term.strip().split("(", 1)[-1].split(",", 1)[0].strip()
        for term in cov.split("+")
        if term.strip()
    ]

    keep = [
        c for c in (response_cols + covariate_list + (list(next(iter(init_kwargs.values()))) if init_kwargs else []))
        if c in adata_bulk.obs.columns
    ]

    if "cell_weight" in adata_bulk.obs.columns:
        raise ValueError("'cell_weight' is a reserved column name. please rename it.")

    # Strict nan-checking as GLMs fail mathematically on missing metadata
    invalid_cols = [col for col in keep if _is_na_strict(adata_bulk.obs[col]).any()]
    if invalid_cols:
        raise ValueError(
            f"bulk metadata contains missing values in columns: {invalid_cols!r}. "
            "strictly fix or remove these samples before modeling."
        )

    bulk_obs = adata_bulk.obs.loc[:, keep].copy()
    bulk_obs["cell_weight"] = 0.0

    logg.info(f"extracting cell weights from `.obsm[{cell_weights_key!r}]`")
    weights_df = adata_sc.obsm[cell_weights_key].loc[:, bulk_obs.index].copy()

    # Cell weights from ridge regression can be infinitesimally small.
    # Passing unscaled, near-zero covariates into the GLM's Newton-Raphson
    # optimizer frequently causes Hessian singularities and convergence failures.
    # We scale by the global maximum to stabilize the solver. Because this is a
    # single linear scalar, the signal-to-noise ratio is perfectly preserved and
    # the resulting GLM p-values are mathematically invariant to the transformation.
    #
    # TODO: Ridge regression leaves "mathematical dust" (e.g., 1e-12) instead of
    # true zeroes. Applying an epsilon-clipping threshold here in the future could
    # increase matrix sparsity and reduce memory overhead.
    if max_normalize_weights:
        max_weight = np.abs(weights_df.to_numpy()).max()
        if max_weight > 0:
            logg.debug(
                f"scaling cell weights by global max ({max_weight:e}) for solver stability"
            )
            weights_df = weights_df / max_weight

    return formula, bulk_obs, weights_df
