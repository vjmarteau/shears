import logging
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from shears.util import _cell_worker_map, _prepare_bulk_obs_and_weights

logg = logging.getLogger(__name__)


def _test_cell_glm(
    cell_weights: np.ndarray,
    endog_array: np.ndarray,
    exog_array: np.ndarray,
    weight_col_idx: int,
    family_class: type,
    init_kwargs: dict[str, Any],
    fit_kwargs: dict[str, Any],
) -> tuple[float, float, float]:
    """Fit a generalized linear model on a single cell using pre-compiled numpy matrices."""
    import warnings

    import numpy as np
    import statsmodels.api as sm

    # Skip cells with zero target variance to prevent c-level optimizers
    # from stalling on perfectly separated clinical arrays.
    mask = cell_weights > 0
    if np.ptp(endog_array[mask]) == 0.0:
        return 1.0, 0.0, float("inf")

    local_family = family_class()

    # Need .copy() to create a private workspace for the hessian inversion
    # to prevent race conditions on the shared memmapped exog array.
    exog_matrix = exog_array.copy()
    exog_matrix[:, weight_col_idx] = cell_weights

    try:
        # Mute c-level floating point overflows (perfect separation) and
        # python-level convergence warnings.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            # Mute all runtime warnings (overflow, divide by zero, invalid multiply) during this block
            warnings.filterwarnings(
                "ignore", 
                category=RuntimeWarning, 
                message="overflow encountered in exp|divide by zero|invalid value encountered"
            )
            
            from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=HessianInversionWarning)

            model = sm.GLM(endog_array, exog_matrix, family=local_family, **init_kwargs)
            fit_result = model.fit(**fit_kwargs)

        # Coerce to raw python scalars to avoid pandas string-index keyerrors downstream
        pval = float(np.asarray(fit_result.pvalues)[weight_col_idx])
        coef = float(np.asarray(fit_result.params)[weight_col_idx])
        se = float(np.asarray(fit_result.bse)[weight_col_idx])
        return pval, coef, se

    except (ValueError, np.linalg.LinAlgError):
        # Fallback for convergence failures like singular matrices or absolute perfect separation
        return 1.0, 0.0, float("inf")


def shears_glm(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    dep_var: str,
    *,
    subset_key: str | None = None,
    family: Any | None = None,
    covariate_str: str | None = None,
    cell_weights_key: str = "cell_weights",
    key_added: str = "shears_glm",
    max_normalize_weights: bool = True,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
    init_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """\
    Compute per-cell association coefficients between single-cell weights and bulk phenotypes.

    Fits a generalized linear model of the form ``dep_var ~ cell_weight + covariates``.

    Parameters
    ----------
    adata_sc
        Annotated single-cell dataset with precomputed cell weights.
    adata_bulk
        Bulk dataset with outcomes and covariates in `.obs`.
    dep_var
        Name of the dependent variable column in `adata_bulk.obs` to model.
    subset_key
        Boolean mask column in `adata_sc.obs` to filter cells prior to GLM fitting.
        If `None`, automatically inherits the subset used during weight generation.
    family
        A statsmodels Family instance for the GLM. Defaults to logistic regression
        (`sm.families.Binomial()`). Passed as `Any` to prevent eager module loading.
    covariate_str
        Optional formulaic-style string of covariates to adjust for.
    cell_weights_key
        Key in `adata_sc.obsm` under which per-cell weights are stored.
    key_added
        Key under which to save the results.
    max_normalize_weights
        Whether to divide the cell weights by their global absolute maximum 
        prior to fitting. Highly recommended to ensure numerical stability 
        in the GLM solver.
    n_jobs
        Number of parallel worker processes.
    chunk_size
        Number of cells to process per chunk in parallel execution.
    init_kwargs
        Keyword arguments passed to the GLM constructor.
    fit_kwargs
        Keyword arguments passed to the GLM fit method.
    inplace
        If `True`, modifies `adata_sc` in place. If `False`, returns the results dataframe.

    Returns
    -------
    If `inplace=True`, returns `None` and saves to `adata_sc.obsm[key_added]`.
    Otherwise, returns the resulting dataframe.
    """
    try:
        import statsmodels.api as sm
        from formulaic import model_matrix
    except ImportError as e:
        raise ImportError("shears_glm requires `statsmodels` and `formulaic`.") from e

    if cell_weights_key not in adata_sc.obsm:
        raise KeyError(f"weights key {cell_weights_key!r} not found in adata_sc.obsm.")

    if family is None:
        family = sm.families.Binomial()
    family_class = type(family)

    if inplace and key_added in adata_sc.uns and key_added in adata_sc.obsm:
        cached_params = adata_sc.uns[key_added].get("params", {})
        if (
            cached_params.get("dep_var") == dep_var
            and cached_params.get("cell_weights_key") == cell_weights_key
            and cached_params.get("covariate_str") == covariate_str
            and cached_params.get("max_normalize_weights") == max_normalize_weights
            and cached_params.get("family") == family_class.__name__
            and cached_params.get("subset_key") == subset_key
        ):
            logg.info(
                f"found cached glm results in `.obsm[{key_added!r}]`. skipping computation."
            )
            return None

    if subset_key is None:
        upstream_params = adata_sc.uns.get(cell_weights_key, {}).get("params", {})
        subset_key = upstream_params.get("subset_key", None)

    n_cells = adata_sc.n_obs
    valid_glm_mask = np.ones(n_cells, dtype=bool)

    if subset_key and subset_key in adata_sc.obs:
        valid_glm_mask &= adata_sc.obs[subset_key].fillna(False).astype(bool)

    # Mathematically, a GLM cannot fit a cell with 0.0 weight across all bulk samples.
    # Dynamically dropping these prevents Hessian collapse and runaway solver times.
    weights_matrix = adata_sc.obsm[cell_weights_key]

    if hasattr(weights_matrix, "sum"):
        cell_weight_sums = np.asarray(weights_matrix.sum(axis=1)).flatten()
    else:
        cell_weight_sums = np.sum(weights_matrix, axis=1)

    has_weight_mask = cell_weight_sums > 0.0
    n_zero_weight_cells = (valid_glm_mask & ~has_weight_mask).sum()

    if n_zero_weight_cells > 0:
        logg.info(f"excluding {n_zero_weight_cells} cells with 0.0 weight across all bulk samples")

    valid_glm_mask &= has_weight_mask

    init_kwargs = init_kwargs or {}

    # Default to quasi-newton L-BFGS optimization. This bypasses exact hessian inversion,
    # preventing the statsmodels solver from hanging on near-perfect separation in sparse arrays.
    fit_kwargs = fit_kwargs or {"method": "lbfgs", "maxiter": 35}

    dep_series = adata_bulk.obs[dep_var]
    if dep_series.dropna().nunique() <= 1:
        raise ValueError(f"target {dep_var!r} has zero variance")

    categories = None
    if isinstance(dep_series.dtype, pd.CategoricalDtype):
        categories = dep_series.cat.categories.tolist()
    elif dep_series.dtype == object or dep_series.nunique() <= 2:
        categories = sorted(dep_series.dropna().unique().tolist())

    if isinstance(family, sm.families.Binomial):
        if categories is None or len(categories) != 2:
            n_cats = len(categories) if categories else "continuous/unrecognized"
            raise ValueError(
                f"binomial family requires exactly 2 categories, but target {dep_var!r} has {n_cats}"
            )

    formula, bulk_obs_df, weights_df = _prepare_bulk_obs_and_weights(
        adata_sc,
        adata_bulk,
        cell_weights_key,
        covariate_str,
        response_cols=[dep_var],
        max_normalize_weights=max_normalize_weights,
    )

    n_filtered = (~valid_glm_mask).sum()
    if n_filtered > 0:
        logg.info(f"subsetting to {valid_glm_mask.sum()} cells")
        weights_df = weights_df.loc[valid_glm_mask].copy()

    if covariate_str:
        logg.info(f"fitting model with formula: {formula}")
    else:
        logg.info("fitting model without covariates.")

    # Convert the formula string to a design matrix once before parallel worker loops
    response_df, predictors_df = model_matrix(formula, bulk_obs_df)

    weight_col_idx = predictors_df.columns.get_loc("cell_weight")

    # formulaic dummy-encodes categorical variables into two columns.
    # grab the second column to ensure positive coefficients point toward the treatment category.
    # the original paper used iloc[:, 0], which flipped signs in some published plots if the
    # reference category sorted first alphabetically. biological magnitude and p-values are unaffected.
    endog_raw = (
        response_df.iloc[:, 1].to_numpy()
        if isinstance(family, sm.families.Binomial) and response_df.shape[1] == 2
        # fallback for continuous variables or already-binary numeric columns
        else response_df.iloc[:, 0].to_numpy()
    )
    exog_raw = predictors_df.to_numpy()

    endog = np.ascontiguousarray(endog_raw, dtype=np.float64)
    exog = np.ascontiguousarray(exog_raw, dtype=np.float64)

    if (
        isinstance(family, sm.families.Binomial)
        and categories is not None
        and len(categories) == 2
    ):
        logg.info(
            f"using {categories[0]!r} as reference and {categories[1]!r} as treatment."
        )

    glm_results_df = _cell_worker_map(
        weights_df,
        _test_cell_glm,
        endog,
        exog,
        weight_col_idx,
        family_class,
        init_kwargs,
        fit_kwargs,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )

    glm_results_df = glm_results_df.fillna({"coef": 0.0, "pvalue": 1.0, "se": np.inf})

    # Pad output matrix with 0 to preserve strict dimensional alignment
    # with the parent AnnData object.
    full_results_df = pd.DataFrame(
        {"coef": 0.0, "pvalue": 1.0, "se": np.inf}, index=adata_sc.obs_names
    )
    full_results_df.loc[glm_results_df.index, glm_results_df.columns] = glm_results_df

    params_dict = {
        "dep_var": dep_var,
        "cell_weights_key": cell_weights_key,
        "covariate_str": covariate_str,
        "max_normalize_weights": max_normalize_weights,
        "subset_key": subset_key,
        "model_type": "glm",
        "family": family_class.__name__,
        "categories": categories,
    }

    if categories is not None and len(categories) == 2:
        params_dict.update(
            {
                "reference": str(categories[0]),
                "treatment": str(categories[1]),
            }
        )

    if inplace:
        adata_sc.obsm[key_added] = full_results_df
        adata_sc.uns[key_added] = {"params": params_dict}
        logg.info(f"added results to `.obsm[{key_added!r}]` and `.uns[{key_added!r}]`")
        return None

    return full_results_df
