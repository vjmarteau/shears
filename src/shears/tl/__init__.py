import functools
import inspect
import itertools
import warnings
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata import AnnData
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceWarning
from shears._util import fdr_correction, process_map
from tqdm import tqdm

# Near-zero variance in cell_weight causes expected convergence warnings when fitting
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# This second warning will only aappear when using the strata argument in lifelines cph.fit and can be remoed when lifelines is updated.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*observed=False is deprecated.*")

# Module-level globals for pre-loaded data
_BULK_OBS = None
_WEIGHTS = None


def _test_cell(cell_name, *, formula, family, duration_col, event_col, init_kwargs, fit_kwargs):

    df = _BULK_OBS.copy(deep=False)
    df["cell_weight"] = _WEIGHTS.loc[cell_name, df.index]

    if family == "cox":
        cph = CoxPHFitter(**init_kwargs)
        cph.fit(
            df,
            duration_col=duration_col,
            event_col=event_col,
            formula=formula,
            **fit_kwargs,
        )
        pval = cph.summary.at["cell_weight", "p"]
        coef = cph.summary.at["cell_weight", "coef"]
    else:
        fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
        res = smf.glm(formula, data=df, family=fam).fit()
        pval = res.pvalues["cell_weight"]
        coef = res.params["cell_weight"]

    return float(pval), float(coef)


def shears(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    *,
    dep_var: str = "",
    covariate_str: str = "",
    inplace: bool = True,
    cell_weights_key: str = "cell_weights",
    key_added: str = "shears",
    family: Literal["binomial", "gaussian", "cox"] = "binomial",
    duration_col: str = "OS_time",
    event_col: str = "OS_status",
    n_jobs: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Compute per-cell association scores between single-cell weights and bulk phenotypes.

    Shears performs a two-step deconvolution and regression procedure:

    1. **Weight estimation**: pre-computed `cell_weights_key` (from `adata_sc.obsm`) deconvolves
       bulk expression profiles (`adata_bulk.obs`) into per-cell contributions via ridge regression.
    2. **Per-cell testing**: for each cell, fit a regression model of the form:
       - GLM for binary (`family="binomial"`) or continuous (`family="gaussian"`) outcomes:
         `dep_var ~ cell_weight + covariates`
       - Cox proportional hazards for survival outcome (`family="cox"`) defined by duration_col and event_col: 
         `cell_weight + covariates`
    The coefficient of `cell_weight` quantifies each cell's association with the bulk phenotype,
    and its p-value measures significance, controlling for additional covariates.

    Parallel execution is enabled via `n_jobs`; set `n_jobs=1` to run single-threaded

    Parameters
    ----------
    adata_sc
        Annotated single-cell data (`AnnData`) with precomputed cell weights in `.obsm[cell_weights_key]`.
    adata_bulk
        Bulk sample metadata (`AnnData`) with outcomes and covariates in `.obs`.
    dep_var
        Dependent variable for GLM families (ignored if `family="cox"`).
    covariate_str
        Additional covariates (e.g. "+ age + sex + batch") inserted into the model formula.
    inplace
        If True, writes the resulting coefficients into `adata_sc.obs[key_added]`.
    key_added
        Column name under which to save per-cell coefficients in `adata_sc.obs`.
    family
        Model type: "binomial", "gaussian", or "cox" for survival analysis.
    duration_col
        Column name in `adata_bulk.obs` for survival durations (when `family="cox"`).
    event_col
        Column name in `adata_bulk.obs` for event indicators (when `family="cox"`).
    n_jobs
        Number of workers for parallel mapping. Use `n_jobs=1` to disable multiprocessing.
    **kwargs
        Additional keyword arguments for the Lifelines CoxPHFitter (e.g., `penalizer`, `l1_ratio`,
        `fit_options`), allowing customization of regularization and convergence settings.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by cell names with columns ["pvalue", "coef"].
    """

    if family == "cox":
        formula = f"cell_weight {covariate_str}"

        init_params = set(inspect.signature(CoxPHFitter.__init__).parameters) - {"self"}
        fit_params = set(inspect.signature(CoxPHFitter.fit).parameters) - {"self"}

        init_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
        fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_params}
        keep = [c for c in adata_bulk.obs.columns if c in (event_col + duration_col + covariate_str + str(fit_kwargs))]

        unused = set(kwargs) - init_kwargs.keys() - fit_kwargs.keys()
        if unused:
            raise TypeError(f"Unexpected CoxPHFitter parameters: {unused}")
    else:
        formula = f"{dep_var} ~ cell_weight {covariate_str}"
        keep = [c for c in adata_bulk.obs.columns if c in (dep_var + covariate_str)]
        init_kwargs = {}
        fit_kwargs  = {}

    assert "cell_weight" not in keep, "cell_weight is reserved"
    print("Formula:", formula)

    bulk_obs = adata_bulk.obs.loc[:, keep].copy()
    weights_df = adata_sc.obsm[cell_weights_key]
    cell_names = list(adata_sc.obs_names)

    # stash into globals so workers inherit them on fork
    global _BULK_OBS, _WEIGHTS
    _BULK_OBS = bulk_obs
    _WEIGHTS = weights_df

    worker = functools.partial(
        _test_cell,
        formula=formula,
        family=family,
        duration_col=duration_col,
        event_col=event_col,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
    )

    res_list = process_map(
        worker,
        cell_names,
        max_workers=n_jobs,
        chunksize=100,
        total=len(cell_names),
        tqdm_class=tqdm,
    )

    df_res = pd.DataFrame(res_list, index=cell_names, columns=["pvalue", "coef"])
    if inplace:
        adata_sc.obs[key_added] = df_res["coef"]
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
