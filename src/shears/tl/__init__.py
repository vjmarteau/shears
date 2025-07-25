import functools
import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from anndata import AnnData
from formulaic import model_matrix
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceWarning
from shears._util import _cell_worker_map, fdr_correction
from statsmodels.genmod.families.family import Family
from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)


def _prepare_bulk_obs_and_weights(
    adata_sc,
    adata_bulk,
    cell_weights_key,
    covariate_str,
    response_cols,
    init_kwargs=None,
):
    """Build the regression formula, subset bulk.obs for required variables, and retrieve per-cell weights."""

    init_kwargs = init_kwargs or {}

    cov = (covariate_str or "").strip(" +")
    covariate_term = f" + {cov}" if cov else ""

    if len(response_cols) == 1:
        formula = f"{response_cols[0]} ~ cell_weight{covariate_term}"
    else:
        formula = f"cell_weight{covariate_term}"

    covariate_list = [
        term.strip().split("(", 1)[-1].split(",", 1)[0].strip()
        for term in cov.split("+")
        if term.strip()
    ]

    keep = [
        c
        for c in (
            response_cols
            + covariate_list
            + (list(next(iter(init_kwargs.values()))) if init_kwargs else [])
        )
        if c in adata_bulk.obs.columns
    ]

    assert "cell_weight" not in adata_bulk.obs.columns, "cell_weight is reserved"

    bulk_obs = adata_bulk.obs.loc[:, keep].copy()
    bulk_obs["cell_weight"] = 0.0

    nan_cols = bulk_obs.columns[bulk_obs.isnull().any()].tolist()
    if nan_cols:
        raise ValueError(f"bulk metadata contains NaNs in columns: {nan_cols!r}")

    weights_df = adata_sc.obsm[cell_weights_key].loc[:, bulk_obs.index]

    return formula, bulk_obs, weights_df


def _test_cell_glm(endog_array, exog_array, cell_weights, family, init_kwargs, fit_kwargs):
    """Compute Wald p-value and coefficient for `cell_weights` by fitting a GLM with the provided family."""

    X = exog_array.copy()
    X[:, -1] = cell_weights

    with threadpool_limits(limits=1):
        res = sm.GLM(endog_array, X, family=family, **init_kwargs).fit(**fit_kwargs)

    return float(res.pvalues[-1]), float(res.params[-1])


def shears_glm(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    dep_var: str,
    *,
    family: Family = sm.families.Binomial(),
    covariate_str: Optional[str] = None,
    cell_weights_key: str = "cell_weights",
    key_added: str = "shears",
    n_jobs: Optional[int] = None,
    inplace: bool = True,
    init_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Compute per-cell association coefficients between single-cell weights and bulk phenotypes fitting a 
    generalized linear model of the form:
       
           dep_var ~ cell_weight + (optional covariates)

    adata_sc
        Annotated single cell AnnData with precomputed cell weights in `.obsm[cell_weights_key]`.
    adata_bulk
        Bulk AnnData with outcomes and covariates in `.obs`.
    dep_var
        Name of the dependent variable column in `adata_bulk.obs` to model.
    family
        A statsmodels `Family` instance such as `sm.families.Binomial()` for logistic regression on binary outcomes or `sm.families.Gaussian()`
         for ordinary least squares on continuous data.
    covariate_str
        Optional formulaic-style strings of covariates to adjust for (e.g. `"age_scaled + C(sex, Treatment(reference='male')) + batch"`).
    cell_weights_key
        Key in `adata_sc.obsm` where per-cell weights are stored.
    key_added
        Column name under which to save the estimated coefficients and pvalues in `adata_sc.obs` if
        `inplace=True`.
    n_jobs
        Number of parallel worker processes. If `None`, uses joblib’s default. Positive values
        set an exact number, `1` disables parallelism, negative values select relative to
        CPU count (e.g., `-1` = all cores).
    inplace
        If True, writes the coefficient and pvalue into `adata_sc.obs[key_added]`.
    init_kwargs
        Optional dict of keyword args passed to `sm.GLM(..., **init_kwargs)`.
    fit_kwargs
        Optional dict of keyword args passed to the `.fit(**fit_kwargs)` call.
    
    Returns
    -------
    Depending on the value of `inplace` either adds two columns to
    `adata.obs` or returns a DataFrame with columns `"coef"` and `"pvalue"`
    """

    init_kwargs = init_kwargs or {}
    fit_kwargs = fit_kwargs or {}

    formula, bulk_obs, weights_df = _prepare_bulk_obs_and_weights(
        adata_sc,
        adata_bulk,
        cell_weights_key,
        covariate_str,
        response_cols=[dep_var],
    )
    logger.debug("Fitting model with formula: %s", formula)

    response_df, predictors_df = model_matrix(formula, bulk_obs)
    endog = response_df.iloc[:, 0].to_numpy()
    exog = predictors_df.to_numpy()

    worker = functools.partial(
        _test_cell_glm,
        endog,
        exog,
        family=family,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
    )
    df_res = _cell_worker_map(weights_df, worker, n_jobs=n_jobs, backend="loky")

    if inplace:
        adata_sc.obs[f"{key_added}_coef"] = df_res["coef"]
        adata_sc.obs[f"{key_added}_pvalue"] = df_res["pvalue"]
    return df_res


def _test_cell_cox(cell_weights, bulk_obs, duration_col, event_col, formula, init_kwargs, fit_kwargs):
    """Fit a lifelines CoxPH model on `bulk_obs` using `cell_weights` and return the Wald p-value and log-hazard coefficient."""

    # Near-zero variance in cell_weight causes expected convergence warnings when fitting
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # This second warning will only appear when using the strata argument in lifelines cph.fit and can be removed when lifelines is updated.
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*observed=False is deprecated.*"
    )

    bulk_obs = bulk_obs.copy()
    bulk_obs["cell_weight"] = cell_weights

    with threadpool_limits(limits=1):
        cph = CoxPHFitter(**init_kwargs)
        cph.fit(
            bulk_obs,
            duration_col=duration_col,
            event_col=event_col,
            formula=formula,
            **fit_kwargs
        )

    pval = cph.summary.at["cell_weight", "p"]
    coef = cph.summary.at["cell_weight", "coef"]
    return float(pval), float(coef)


def shears_cox(
    adata_sc,
    adata_bulk,
    duration_col: str = "OS_time",
    event_col: str = "OS_status",
    *,
    covariate_str: Optional[str] = None,
    cell_weights_key: str = "cell_weights",
    key_added: str = "shears",
    n_jobs: Optional[int] = None,
    inplace: bool = True,
    init_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Compute per-cell association coefficients between single-cell weights and time-to-event outcomes
    by fitting a Cox proportional hazards model using lifelines.CoxPHFitter.

    Parameters
    ----------
    adata_sc
        Annotated single cell AnnData with precomputed cell weights in `.obsm[cell_weights_key]`.
    adata_bulk
        Bulk AnnData object with survival data and covariates in `.obs`.
    duration_col
        Column name in `adata_bulk.obs` containing follow-up or survival time.
    event_col
        Column name in `adata_bulk.obs` containing event indicator (1=event occurred, 0=censored).
    covariate_str
        Optional formulaic-style strings of covariates to adjust for (e.g. `"age_scaled + C(sex, Treatment(reference='male')) + batch"`).
    cell_weights_key
        Key in `adata_sc.obsm` under which per-cell weights are stored.
    key_added
        Column name under which to save the estimated coefficients in `adata_sc.obs` if `inplace=True`.
    n_jobs
        Number of parallel worker processes. If `None`, uses joblib’s default. Positive values
        set an exact number, `1` disables parallelism, negative values select relative to
        CPU count (e.g., `-1` = all cores).
    inplace
        If `True`, writes the `"coef"` and `"pvalue"` columns into `adata_sc.obs[key_added]`.
    init_kwargs
        Additional keyword arguments passed to the Cox model constructor.
    fit_kwargs
        Additional keyword arguments passed to the `.fit(**fit_kwargs)` call.

    Returns
    -------
    DataFrame
       Depending on the value of `inplace` either adds two columns to `adata.obs` or returns a DataFrame with columns `"coef"` and `"pvalue"`
    """

    init_kwargs = init_kwargs or {}
    fit_kwargs = fit_kwargs or {}

    formula, bulk_obs, weights_df = _prepare_bulk_obs_and_weights(
        adata_sc,
        adata_bulk,
        cell_weights_key,
        covariate_str,
        response_cols=[duration_col, event_col],
        init_kwargs=init_kwargs,
    )
    logger.debug("Fitting model with formula: %s", formula)

    worker = functools.partial(
        _test_cell_cox,
        bulk_obs=bulk_obs,
        duration_col=duration_col,
        event_col=event_col,
        formula=formula,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
    )
    df_res = _cell_worker_map(weights_df, worker, n_jobs=n_jobs, backend="loky")

    if inplace:
        adata_sc.obs[f"{key_added}_coef"] = df_res["coef"]
        adata_sc.obs[f"{key_added}_pvalue"] = df_res["pvalue"]
    return df_res


def get_scaling_factor(adata_sc, cell_type_col, *, weight_col="n_genes", callback=np.median):
    """Get scaling factors per cell-type.

    Corrects for the number of cells per dataset and the mRNA content of cells.

    Parameters
    ----------
    adata_sc
        single cell AnnData
    cell_type_col
        column in adata_sc.obs that holds cell-type information
    weight_col
        column in adata.obs that holds a cell weight to correc for mRNA bias. Viable options are
        the number of detected genese or the number of total counts per cell. See Dietrich et al. for
        more details
    callback
        Aggregate the values in weight_col by cell_type using this function.
    """
    # multiply weights with the number of cells (if there are many cells of a type in the single-cell dataset,
    # the factors will distribute across more cells and the weights dilute)
    n_cells = adata_sc.obs[cell_type_col].value_counts()

    # divide by a scaling factor to account for mRNA contents. Gene expression is normalized,
    # therefore e.g. Neutrophils (low mRNA) and Macrophages (high mRNA) have the same amount of normalized counts.
    # However, if they get the same weight, there will be more Neutrophils in the bulk sample,
    # because they contribute less mRNA to the bulk sample.
    cell_weight = adata_sc.obs[weight_col].groupby(adata_sc.obs[cell_type_col]).agg(callback)

    factor = n_cells / cell_weight
    return factor.reindex(adata_sc.obs[cell_type_col])


def cell_type_fractions(adata_sc, cell_type_col, as_fraction=True, bias_correction=True):
    """Compute cell-type fractions from cell weights"""
    cell_weights = adata_sc.obsm["cell_weights"]

    if bias_correction:
        cell_weights = cell_weights.multiply(get_scaling_factor(adata_sc, cell_type_col).values, axis=0)

    df = cell_weights.groupby(adata_sc.obs[cell_type_col]).agg(sum)

    if as_fraction:
        df = df.div(df.sum(axis=0).to_dict())

    return df


def shears_stats(adata_sc, res, groupby, batch_key: str, cell_cutoff=20):
    """Compute statistics on shears results"""
    df = (
        res.assign(weight=lambda x: x["coef"] * -np.log10(x["pvalue"]))
        .assign(group=adata_sc.obs[groupby], replicate=adata_sc.obs[batch_key])
        .merge(
            adata_sc.obs.groupby(groupby, observed=True)[batch_key]
            .value_counts()
            .reset_index(name="n_cells"),
            left_on=["group", "replicate"],
            right_on=[groupby, batch_key],
            how="left"
        )
        .drop(columns=[groupby, batch_key])
        # only consider samples with at least cell_cutoff cells before aggregating by sample
        .loc[lambda x: x['n_cells'] >= cell_cutoff]
        .groupby(["group", "replicate"], observed=True)
        .agg(mean_weight=("weight", "mean"))
        .reset_index()
    )
    df = df.merge(df["group"].value_counts().reset_index(name="n_samples"), on="group", how="left")
    
    stats_df = (
        df.groupby("group", observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    "median_weight": np.median(x["mean_weight"]),
                    # **scipy.stats.ttest_1samp(x["mean_weight"], 0)._asdict(),
                    **scipy.stats.wilcoxon(x["mean_weight"])._asdict(),
                }
            ),
            include_groups=False
        )
        .reset_index()
        .pipe(fdr_correction)
        .sort_values("pvalue")
    )
    return df.merge(stats_df, on="group", how="left")
