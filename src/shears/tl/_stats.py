import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats
from anndata import AnnData
from shears.util import fdr_correction
from shears.util._wrangling import _get_clean_mask

logg = logging.getLogger(__name__)


def _wilcoxon_rank_sum(
    abundance_df: pd.DataFrame,
    groupby: str,
    metric_col: str,
) -> pd.DataFrame:
    """Perform Wilcoxon rank-sum tests to evaluate differential signature abundance across groups."""
    group_median_col = f"group_{metric_col}"

    def _safe_wilcoxon(group_df: pd.DataFrame) -> pd.Series:
        metric_values = group_df[metric_col].dropna().to_numpy()

        # scipy.stats.wilcoxon requires at least 2 observations to compute differences.
        # It also raises an error if all differences are zero. We intercept these cases
        # and assign a p-value of 1.0, as they contain zero differential signal.
        if len(metric_values) < 2 or np.ptp(metric_values) == 0.0:
            return pd.Series(
                {
                    group_median_col: (
                        float(np.median(metric_values))
                        if len(metric_values) > 0
                        else np.nan
                    ),
                    "statistic": 0.0,
                    "pvalue": 1.0,
                }
            )

        group_median = float(np.median(metric_values))

        # scipy throws UserWarnings when N < 10 or when exact p-values cannot be
        # computed due to ties. We silence these because low-N comparisons and ties
        # are mathematically valid and practically unavoidable in biological replicates.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wilcoxon_res = scipy.stats.wilcoxon(metric_values)

        return pd.Series(
            {
                group_median_col: group_median,
                "statistic": float(wilcoxon_res.statistic),
                "pvalue": float(wilcoxon_res.pvalue),
            }
        )

    return (
        abundance_df.groupby(groupby, observed=True)
        .apply(_safe_wilcoxon, include_groups=False)
        .reset_index()
        .pipe(fdr_correction)
        .sort_values("pvalue")
    )


def _aggregate_wald_stats(
    glm_results_df: pd.DataFrame,
    cell_metadata_df: pd.DataFrame,
    groupby: str,
    batch_key: str,
    min_cells: int,
    se_cutoff: float,
    scaling_factor_col: str | None,
) -> pd.DataFrame:
    """\
    Exclude unstable GLM fits and aggregate single-cell deconvolution weights into biological replicates for differential testing.
    """
    # Clip the standard error to prevent division-by-zero errors and penalize mathematically
    # unstable GLM fits when calculating the Wald statistic.
    stable_se = np.clip(glm_results_df["se"].to_numpy(), 1e-6, None)
    glm_results_df["wald_stat"] = glm_results_df["coef"] / stable_se

    if scaling_factor_col is not None:
        glm_results_df["coef_scaled"] = (
            glm_results_df["coef"] / cell_metadata_df[scaling_factor_col].values
        )

        # The Wald statistic is mathematically invariant to scalar division.
        # Duplicated strictly to satisfy downstream API expectations for bias-corrected queries.
        glm_results_df["wald_stat_scaled"] = glm_results_df["wald_stat"]
        target_metric = "median_wald_stat_scaled"
    else:
        target_metric = "median_wald_stat"

    # Define convergence boundaries, mathematically a fit is stable only if its standard
    # error is constrained and its effect size is biologically plausible.
    convergence_mask = np.ones(len(glm_results_df), dtype=bool)
    if "se" in glm_results_df.columns:
        convergence_mask = (glm_results_df["se"].to_numpy() <= se_cutoff) & (
            glm_results_df["coef"].abs().to_numpy() < 25.0
        )

    # Track convergence metadata to inform visual thresholding in downstream plotters.
    group_convergence_df = (
        pd.DataFrame(
            {
                groupby: cell_metadata_df[groupby].values,
                "is_converged": convergence_mask,
            }
        )
        .groupby(groupby, observed=True)
        .agg(
            total_cells=("is_converged", "count"),
            converged_cells=("is_converged", "sum"),
        )
        .assign(convergence_rate=lambda x: x["converged_cells"] / x["total_cells"])
        .reset_index()
    )

    # Mask degenerate fits (ill-conditioned Hessians or runaway coefficients)
    # to prevent spurious outliers from dominating the group median calculation.
    unstable_fit_mask = ~convergence_mask
    n_unstable_cells = unstable_fit_mask.sum()
    if n_unstable_cells > 0:
        logg.info(
            f"masking weights and coefficients for {n_unstable_cells} cells (failed convergence)."
        )
        glm_results_df.loc[unstable_fit_mask, ["coef", "weight", "wald_stat"]] = np.nan
        if scaling_factor_col is not None:
            glm_results_df.loc[
                unstable_fit_mask, ["coef_scaled", "weight_scaled", "wald_stat_scaled"]
            ] = np.nan

    # Dynamically filter out pseudobulk groups lacking sufficient single-cell support
    # to prevent high-variance artifacts.
    annotated_weights_df = glm_results_df.assign(
        **{
            groupby: cell_metadata_df[groupby].values,
            batch_key: cell_metadata_df[batch_key].values,
        }
    )
    cells_per_batch_df = (
        annotated_weights_df.groupby([groupby, batch_key], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    annotated_weights_df = annotated_weights_df.merge(
        cells_per_batch_df, on=[groupby, batch_key], how="left"
    )

    sufficient_cells_mask = annotated_weights_df["n_cells"] >= min_cells
    n_underpowered_groups = (
        annotated_weights_df.loc[~sufficient_cells_mask, [groupby, batch_key]]
        .drop_duplicates()
        .shape[0]
    )

    if n_underpowered_groups > 0:
        logg.info(
            f"filtered out {n_underpowered_groups} {groupby!r} groups across {batch_key!r} with < {min_cells} cells."
        )

    aggregation_funcs = {
        "median_coef": ("coef", "median"),
        "median_wald_stat": ("wald_stat", "median"),
    }
    if scaling_factor_col is not None:
        aggregation_funcs["median_coef_scaled"] = ("coef_scaled", "median")
        aggregation_funcs["median_wald_stat_scaled"] = ("wald_stat_scaled", "median")

    batch_aggregated_df = (
        annotated_weights_df[sufficient_cells_mask]
        .groupby([groupby, batch_key], observed=True)
        .agg(**aggregation_funcs)
        .reset_index()
    )

    # Ensure statistical validity by requiring at least two distinct biological
    # replicates per condition before passing arrays to the non-parametric tests.
    biological_replicates_df = (
        batch_aggregated_df.groupby(groupby, observed=True)
        .size()
        .reset_index(name="n_samples")
    )
    batch_aggregated_df = batch_aggregated_df.merge(
        biological_replicates_df, on=groupby, how="left"
    )

    insufficient_replicates_mask = batch_aggregated_df["n_samples"] < 2
    if insufficient_replicates_mask.any():
        for grp in batch_aggregated_df.loc[
            insufficient_replicates_mask, groupby
        ].unique():
            logg.info(f"removed {grp!r} (n < 2 biological replicates).")
        batch_aggregated_df = batch_aggregated_df[~insufficient_replicates_mask].copy()

    wilcoxon_results_df = _wilcoxon_rank_sum(
        batch_aggregated_df, groupby, target_metric
    )

    return batch_aggregated_df.merge(wilcoxon_results_df, on=groupby, how="left").merge(
        group_convergence_df, on=groupby, how="left"
    )


def differential_composition(
    adata_sc: AnnData,
    *,
    model_key: str = "shears_glm",
    groupby: str = "cell_type",
    batch_key: str = "patient",
    key_added: str = "shears_stats",
    min_cells: int = 20,
    max_se: float = 25.0,
    scaling_key: str | None = "mRNA_scaling_factor",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """\
    Compute differential composition statistics on single-cell deconvolution results.

    This function aggregates per-cell regression coefficients into biological replicates
    and evaluates significance using the Wald statistic. 

    **Biological Interpretation of the Wald Statistic:**
    The Wald statistic represents the ratio of the biological effect size to its
    mathematical uncertainty ($Z = \\frac{\\beta}{SE}$). It acts as a signal-to-noise ratio:
    * A score near 0 means the model is entirely unsure; the signal and noise cancel out.
    * A score of $\\pm1.96$ indicates the signal is roughly twice the noise, corresponding 
      to a ~95% statistical confidence threshold.
    * This metric naturally penalizes noisy, low-count populations that exhibit massive 
      effect sizes but extreme standard errors.

    Parameters
    ----------
    adata_sc
        Annotated data matrix containing deconvolution outputs.
    model_key
        Key in `adata_sc.uns` and `adata_sc.obsm` where GLM model results are stored.
    groupby
        Column in `adata_sc.obs` defining the primary biological groups to compare.
    batch_key
        Column in `adata_sc.obs` defining biological replicates or batches (e.g., patients).
    key_added
        Key in `adata_sc.uns` under which to save the statistical results.
    min_cells
        Minimum number of cells required per group/batch combination to be included.
        Groups with fewer cells are dynamically dropped to prevent high-variance artifacts.
    max_se
        Maximum allowable standard error. Cells exceeding this threshold are flagged as 
        degenerate fits (e.g., corrupted Hessians) and are masked from downstream aggregation.
    scaling_key
        Column in `adata_sc.obs` containing scaling factors for biological bias correction.
        Set to `None` to evaluate raw, unscaled coefficients.
    inplace
        If `True`, saves results directly to `adata_sc.uns[key_added]`. If `False`, 
        returns the results as a pandas DataFrame.

    Returns
    -------
    If `inplace` is `True`, returns `None`.
    Otherwise, returns the summary DataFrame.
    """
    if model_key not in adata_sc.obsm or model_key not in adata_sc.uns:
        raise KeyError(
            f"model key {model_key!r} not found in adata_sc.obsm or adata_sc.uns."
        )

    scaling_key = None if scaling_key == "None" else scaling_key

    required_metadata_cols = [groupby, batch_key]
    if scaling_key:
        if scaling_key not in adata_sc.obs:
            raise KeyError(f"scaling factor column {scaling_key!r} not found.")
        required_metadata_cols.append(scaling_key)
        logg.info(f"applying biological bias correction using {scaling_key!r}.")

    # Dynamically remove missing metadata (NaN, None, "Unknown") from the arrays
    # to prevent silent dropping or grouping crashes in the downstream pandas merges.
    clean_metadata_mask = _get_clean_mask(adata_sc, required_metadata_cols)

    model_params = adata_sc.uns[model_key].get("params", {})
    upstream_subset_key = model_params.get("subset_key", None)
    
    if upstream_subset_key and upstream_subset_key in adata_sc.obs:
        upstream_mask = adata_sc.obs[upstream_subset_key].fillna(False).astype(bool)

        n_ghost_cells = (~upstream_mask & clean_metadata_mask).sum()
        if n_ghost_cells > 0:
            logg.info(f"excluding {n_ghost_cells} cells omitted by upstream subset {upstream_subset_key!r}.")
        
        clean_metadata_mask &= upstream_mask

    glm_results_df = adata_sc.obsm[model_key][clean_metadata_mask].copy()
    cell_metadata_df = adata_sc.obs.loc[
        clean_metadata_mask, required_metadata_cols
    ].copy()

    cell_metadata_df[groupby] = cell_metadata_df[groupby].astype("category")
    cell_metadata_df[batch_key] = cell_metadata_df[batch_key].astype("category")

    differential_results_df = _aggregate_wald_stats(
        glm_results_df=glm_results_df,
        cell_metadata_df=cell_metadata_df,
        groupby=groupby,
        batch_key=batch_key,
        min_cells=min_cells,
        se_cutoff=max_se,
        scaling_factor_col=scaling_key,
    )

    if inplace:
        adata_sc.uns[key_added] = {
            "results": differential_results_df,
            "params": {
                "groupby": groupby,
                "batch_key": batch_key,
                "min_cells": min_cells,
                "max_se": max_se,
                "scaling_key": scaling_key,
                "model_key": model_key,
                "model_params": adata_sc.uns[model_key].get("params", {}),
                "metric": (
                    "median_wald_stat_scaled" if scaling_key else "median_wald_stat"
                ),
            },
        }
        logg.info(f"added results to `.uns[{key_added!r}]`")
        return None

    return differential_results_df
