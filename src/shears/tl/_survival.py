import logging
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from shears.util import _cell_worker_map, _prepare_bulk_obs_and_weights

logg = logging.getLogger(__name__)


def _test_cell_cox(
    cell_weights: np.ndarray,
    bulk_df_template: pd.DataFrame,
    duration_col: str,
    event_col: str,
    init_kwargs: dict[str, Any],
    fit_kwargs: dict[str, Any],
) -> tuple[float, float, float]:
    """Fit a lifelines coxph model on bulk_obs using cell_weights."""
    import warnings

    # Import inside the worker to prevent joblib from serializing massive classes!
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError

    # Shallow copy assignment updates the weights column
    # without duplicating the underlying dataframe memory
    local_df = bulk_df_template.assign(cell_weight=cell_weights)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit natively on the numeric dataframe, entirely bypassing slow string formulas
            cph = CoxPHFitter(**init_kwargs).fit(
                local_df,
                duration_col=duration_col,
                event_col=event_col,
                **fit_kwargs,
            )

        pval = float(cph.summary.at["cell_weight", "p"])
        coef = float(cph.summary.at["cell_weight", "coef"])
        se = float(cph.summary.at["cell_weight", "se(coef)"])

    except Exception:
        pval, coef, se = 1.0, 0.0, float("inf")

    return pval, coef, se


def shears_cox(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    duration_col: str = "OS_time",
    event_col: str = "OS_status",
    *,
    subset_key: str | None = None,
    covariate_str: str | None = None,
    cell_weights_key: str = "cell_weights",
    key_added: str = "shears_cox",
    max_normalize_weights: bool = True,
    chunk_size: int | None = None,
    n_jobs: int | None = None,
    init_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """\
    Compute per-cell association coefficients between single-cell weights and time-to-event outcomes.

    Fits a Cox proportional hazards model using `lifelines.CoxPHFitter`.

    Parameters
    ----------
    adata_sc
        Annotated single-cell dataset with precomputed cell weights.
    adata_bulk
        Bulk dataset with survival data and covariates in `.obs`.
    duration_col
        Column name in `adata_bulk.obs` containing follow-up or survival time.
    event_col
        Column name in `adata_bulk.obs` containing event indicator (1=event occurred, 0=censored).
    subset_key
        Boolean mask column in `adata_sc.obs` to filter cells prior to fitting.
        If `None`, automatically inherits the subset used during weight generation.
    covariate_str
        Optional formulaic-style string of covariates to adjust for.
    cell_weights_key
        Key in `adata_sc.obsm` under which per-cell weights are stored.
    key_added
        Key under which to save the results.
    max_normalize_weights
        Whether to divide the cell weights by their global absolute maximum 
        prior to fitting. Highly recommended to ensure numerical stability.
    chunk_size
        Number of cells to process per chunk in parallel execution.
    n_jobs
        Number of parallel worker processes.
    init_kwargs
        Keyword arguments passed to the CoxPHFitter constructor.
    fit_kwargs
        Keyword arguments passed to the CoxPHFitter fit method.
    inplace
        If `True`, modifies `adata_sc` in place. If `False`, returns the results dataframe.

    Returns
    -------
    If `inplace=True`, returns `None` and saves to `adata_sc.obsm[key_added]`.
    Otherwise, returns the resulting dataframe.
    """
    try:
        import lifelines  # noqa: F401
        from formulaic import model_matrix
    except ImportError as e:
        raise ImportError("shears_cox requires `lifelines` and `formulaic`.") from e

    if inplace and key_added in adata_sc.uns and key_added in adata_sc.obsm:
        cached_params = adata_sc.uns[key_added].get("params", {})
        if (
            cached_params.get("duration_col") == duration_col
            and cached_params.get("event_col") == event_col
            and cached_params.get("cell_weights_key") == cell_weights_key
            and cached_params.get("covariate_str") == covariate_str
            and cached_params.get("max_normalize_weights") == max_normalize_weights
            and cached_params.get("subset_key") == subset_key
        ):
            logg.info(
                f"found cached cox results in `.obsm[{key_added!r}]`. skipping computation."
            )
            return None

    if subset_key is None:
        upstream_params = adata_sc.uns.get(cell_weights_key, {}).get("params", {})
        subset_key = upstream_params.get("subset_key", None)

    n_cells = adata_sc.n_obs
    valid_cox_mask = np.ones(n_cells, dtype=bool)

    if subset_key and subset_key in adata_sc.obs:
        valid_cox_mask &= adata_sc.obs[subset_key].fillna(False).astype(bool)

    weights_matrix = adata_sc.obsm[cell_weights_key]

    if hasattr(weights_matrix, "sum"):
        cell_weight_sums = np.asarray(weights_matrix.sum(axis=1)).flatten()
    else:
        cell_weight_sums = np.sum(weights_matrix, axis=1)

    has_weight_mask = cell_weight_sums > 0.0
    n_zero_weight_cells = (valid_cox_mask & ~has_weight_mask).sum()

    if n_zero_weight_cells > 0:
        logg.info(
            f"excluding {n_zero_weight_cells} cells with sum(weight) == 0.0 "
            "to prevent mathematically degenerate Cox fits."
        )

    valid_cox_mask &= has_weight_mask

    init_kwargs = init_kwargs or {}
    fit_kwargs = fit_kwargs or {}

    # Handle complete separation and the Hauck-Donner effect.
    # Single-cell data sparsity frequently leads to complete separation, where a 
    # rare cell state perfectly predicts an outcome. This can cause the optimizer 
    # to yield infinitely large coefficients. While standard errors (SE) typically 
    # approach infinity as well, the optimizer may stall early, resulting in a 
    # large coefficient with a finite SE. This can bypass downstream `se_cutoff` 
    # filters and skew patient-level median aggregations.
    #
    # To mitigate this, a default L2 (ridge) penalty is applied to the Cox model. 
    # This ensures an invertible Hessian and stabilizes the fits.
    #
    # Note on GLMs: `statsmodels.GLM.fit_regularized()` intentionally suppresses 
    # standard errors and p-values. This aligns with strict statistical principles 
    # regarding post-selection inference, which argue that calculating standard 
    # errors for artificially shrunk coefficients is invalid. While packages like 
    # `lifelines` compute standard errors for penalized models as a pragmatic 
    # exception, `statsmodels` enforces this theoretical restriction. Because 
    # downstream functions require the SE to compute the Wald statistic and p-values 
    # for FDR correction, GLMs must be fit unpenalized. Unstable GLM fits are 
    # instead handled via a strict `se_cutoff` during aggregation.
    if "penalizer" not in init_kwargs:
        init_kwargs["penalizer"] = 1e-4

    formula_string, bulk_metadata_df, weights_df = _prepare_bulk_obs_and_weights(
        adata_sc,
        adata_bulk,
        cell_weights_key,
        covariate_str,
        response_cols=[duration_col, event_col],
        max_normalize_weights=max_normalize_weights,
        init_kwargs=init_kwargs,
    )

    n_filtered = (~valid_cox_mask).sum()
    if n_filtered > 0:
        logg.info(f"subsetting to {valid_cox_mask.sum()} mathematically valid cells.")
        weights_df = weights_df.loc[valid_cox_mask].copy()

    if covariate_str:
        logg.info(f"fitting model with formula: {formula_string}")
    else:
        logg.info("fitting model without covariates.")

    rhs_formula = formula_string if "~" in formula_string else f"~ {formula_string}"
    covariate_design_df = model_matrix(rhs_formula, bulk_metadata_df)

    if "Intercept" in covariate_design_df.columns:
        covariate_design_df = covariate_design_df.drop(columns=["Intercept"])

    covariate_design_df = covariate_design_df.astype(np.float64)

    extra_cols = []
    if init_kwargs:
        if "strata" in init_kwargs:
            strata = init_kwargs["strata"]
            extra_cols.extend([strata] if isinstance(strata, str) else strata)
        if "cluster_col" in init_kwargs:
            extra_cols.append(init_kwargs["cluster_col"])

    extra_cols = [c for c in extra_cols if c in bulk_metadata_df.columns and c not in covariate_design_df.columns]

    df_parts = [bulk_metadata_df[[duration_col, event_col]]]
    if extra_cols:
        df_parts.append(bulk_metadata_df[extra_cols])
    df_parts.append(covariate_design_df)

    cox_input_template_df = pd.concat(df_parts, axis=1)
    cox_input_template_df["cell_weight"] = 0.0

    cox_results_df = _cell_worker_map(
        weights_df,
        _test_cell_cox,
        cox_input_template_df,
        duration_col,
        event_col,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        backend="loky",
    )

    cox_results_df = cox_results_df.fillna({"coef": 0.0, "pvalue": 1.0, "se": np.inf})

    # Pad output matrix with 0 to preserve strict dimensional alignment
    # with the parent AnnData object.
    full_results_df = pd.DataFrame(
        {"coef": 0.0, "pvalue": 1.0, "se": np.inf}, index=adata_sc.obs_names
    )
    full_results_df.loc[cox_results_df.index, cox_results_df.columns] = cox_results_df

    params_dict = {
        "duration_col": duration_col,
        "event_col": event_col,
        "cell_weights_key": cell_weights_key,
        "covariate_str": covariate_str,
        "max_normalize_weights": max_normalize_weights,
        "subset_key": subset_key,
        "model_type": "cox",
        "fitter": "CoxPHFitter",
    }

    if inplace:
        adata_sc.obsm[key_added] = full_results_df
        adata_sc.uns[key_added] = {"params": params_dict}
        logg.info(f"added results to `.obsm[{key_added!r}]` and `.uns[{key_added!r}]`")
        return None

    return full_results_df
