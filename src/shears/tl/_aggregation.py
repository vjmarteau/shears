import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy import sparse
from shears.util._wrangling import _get_clean_mask

logg = logging.getLogger(__name__)


def calculate_scaling_factors(
    adata_sc: AnnData,
    groupby: str,
    batch_key: str | None = None,
    *,
    weight_col: str = "n_genes",
    callback: str = "median",
    clip_fraction: float = 0.05,
    key_added: str = "mRNA_scaling_factor",
    copy: bool = False,
) -> AnnData | None:
    """\
    Calculate scaling factors per group to correct for mRNA content and cell prevalence.

    Parameters
    ----------
    adata_sc
        Annotated data matrix.
    groupby
        Column in `adata_sc.obs` that holds group/cell-type information.
    batch_key
        Optional column in `adata_sc.obs` for batch-aware scaling (e.g., platform or dataset).
    weight_col
        Column in `adata_sc.obs` that holds a cell weight to correct for mRNA bias. Viable options are
        the number of detected genes or the number of total counts per cell.
    callback
        Aggregate the values in `weight_col` by `groupby` using this function (e.g., "median", "mean").
    clip_fraction
        Fraction of extreme scaling factors to clip (winsorize) at the top and bottom to prevent
        variance explosion. Set to 0.0 to disable.
    key_added
        Column name under which to save the scaling factors in `adata_sc.obs`.
    copy
        If `True`, returns a copied `AnnData` object instead of modifying in place.

    Returns
    -------
    Returns the modified `AnnData` object if `copy=True`, otherwise returns `None`.
    """
    if copy:
        adata_sc = adata_sc.copy()

    if weight_col not in adata_sc.obs.columns:
        raise KeyError(
            f"Column {weight_col!r} is missing from `adata_sc.obs`. "
            f"To generate it, run `sc.pp.filter_cells(adata_sc, min_genes=1)`."
        )

    group_cols = [groupby] if batch_key is None else [batch_key, groupby]
    grouped = adata_sc.obs.groupby(group_cols, observed=True)[weight_col]

    # multiply weights with the number of cells so that abundant types
    # distribute their scaling factors correctly without diluting the effect.
    n_cells = grouped.transform("count")

    # divide by a scaling factor to account for mRNA contents. gene expression is normalized,
    # therefore e.g. neutrophils (low mRNA) and macrophages (high mRNA) have the same amount of
    # normalized counts. if they get the same weight, there will be artificially more neutrophils
    # in the bulk sample, because they actually contribute less mRNA.
    cell_weight = grouped.transform(callback)

    scaling_factors = n_cells / cell_weight

    if clip_fraction > 0.0:
        lower_bound = float(scaling_factors.quantile(clip_fraction))
        upper_bound = float(scaling_factors.quantile(1.0 - clip_fraction))

        scaling_factors = scaling_factors.clip(lower=lower_bound, upper=upper_bound)
        logg.info(
            f"clipped scaling factors beyond the {clip_fraction:.0%} and {1-clip_fraction:.0%} quantiles"
        )

    adata_sc.obs[key_added] = scaling_factors
    logg.info(f"added {key_added!r} to `.obs`")

    return adata_sc if copy else None


def scale_obsm(
    adata_sc: AnnData,
    obsm_key: str,
    scaling_factor_col: str,
    *,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Scale an obsm matrix by a cell-level biological bias factor.

    Parameters
    ----------
    adata_sc
        Annotated single-cell dataset.
    obsm_key
        Key in `adata_sc.obsm` to scale.
    scaling_factor_col
        Column in `adata_sc.obs` containing scaling factors.
    key_added
        Key under which to save the scaled results in `adata_sc.obsm`. 
        Defaults to `{obsm_key}_scaled`.
    copy
        If `True`, returns a copied `AnnData` object instead of modifying in place.

    Returns
    -------
    Returns the modified `AnnData` object if `copy=True`, otherwise returns `None`.
    """
    if copy:
        adata_sc = adata_sc.copy()

    if obsm_key not in adata_sc.obsm:
        raise KeyError(f"Key {obsm_key!r} not found in `adata_sc.obsm`.")
    if scaling_factor_col not in adata_sc.obs:
        raise KeyError(f"Column {scaling_factor_col!r} not found in `adata_sc.obs`.")

    obsm_data = adata_sc.obsm[obsm_key]
    weights = adata_sc.obs[scaling_factor_col].values[:, None]

    if sparse.issparse(obsm_data):
        res_cell = obsm_data.multiply(weights)
    else:
        res_cell = obsm_data * weights

    k = key_added or f"{obsm_key}_scaled"
    adata_sc.obsm[k] = res_cell
    logg.info(f"added scaled matrix {k!r} to `.obsm`")

    return adata_sc if copy else None


def _aggregate_sparse(
    obsm_array: np.ndarray | sp.spmatrix,
    group_idx: np.ndarray,
    cells_per_group: np.ndarray,
    scaling_weights: np.ndarray,
    n_groups: int,
    agg_type: str,
) -> np.ndarray:
    """\
    Core mathematical engine for fast group aggregation.
    Abstracted from AnnData to enforce pure numpy/scipy algebraic optimizations.
    """
    if agg_type == "mean":
        # add epsilon to denominator to handle ZeroDivisionError
        # on empty or highly sparse categorical groups
        scaling_weights = scaling_weights / (
            cells_per_group[group_idx] + np.finfo(float).eps
        )

    # map cells to groups via sparse dot product
    indicator = sp.csr_matrix(
        (scaling_weights, (np.arange(len(group_idx)), group_idx)),
        shape=(len(group_idx), n_groups),
    )

    return indicator.T @ obsm_array


def aggregate_obsm(
    adata_sc: AnnData,
    *,
    obsm_key: str = "cell_weights",
    groupby: str | Sequence[str],
    subset_key: str | None = None,
    min_cells: int = 20,
    agg_type: str = "mean",
    scaling_factor_col: str | None = None,
) -> AnnData:
    """\
    Aggregate an obsm matrix by one or more categorical groupings and return a pseudobulk AnnData.

    Parameters
    ----------
    adata_sc
        Annotated single-cell dataset.
    obsm_key
        Key in `adata_sc.obsm` to aggregate.
    groupby
        Column name or list of column names in `adata_sc.obs` defining the primary biological groups.
    subset_key
        Optional key in `adata_sc.obs` containing a boolean mask of cells to strictly include.
    min_cells
        Minimum number of cells required per group to prevent noisy aggregations.
    agg_type
        Type of aggregation (`'mean'` or `'sum'`).
    scaling_factor_col
        Optional column in `adata_sc.obs` containing global scaling weights.

    Returns
    -------
    A new aggregated `AnnData` object where `obs_names` correspond to the categories in `groupby`.
    """
    if obsm_key not in adata_sc.obsm:
        raise KeyError(f"key {obsm_key!r} not found in `adata_sc.obsm`.")
    if agg_type not in ["mean", "sum"]:
        raise ValueError("agg_type must be 'mean' or 'sum'.")

    # Handle single vs multiple groupby columns natively
    if isinstance(groupby, str):
        group_cols = [groupby]
    else:
        group_cols = list(groupby)

    for col in group_cols:
        if col not in adata_sc.obs:
            raise KeyError(f"groupby column {col!r} not found in `adata_sc.obs`.")

    # aggressive masking: silently drop NaNs across all requested grouping columns
    valid_mask = ~adata_sc.obs[group_cols].isna().any(axis=1).to_numpy(dtype=bool)

    if subset_key is not None:
        if subset_key not in adata_sc.obs:
            logg.warning(f"subset_key {subset_key!r} not found in `.obs`. ignoring")
        else:
            valid_mask &= adata_sc.obs[subset_key].to_numpy(dtype=bool)
            logg.info(f"subsetting aggregation to {valid_mask.sum()} valid cells")

    obs_valid = adata_sc.obs.iloc[valid_mask].copy()
    obsm_raw = adata_sc.obsm[obsm_key][valid_mask]
    obsm_array = obsm_raw.to_numpy() if hasattr(obsm_raw, "to_numpy") else obsm_raw

    # optimize grouping using native pandas categorical codes to map directly to sparse matrix
    if len(group_cols) == 1:
        group_series = obs_valid[group_cols[0]].astype("category")
    else:
        # Create a unified string identifier for multi-level grouping (e.g., "T-cell_Patient1")
        group_series = (
            obs_valid[group_cols]
            .astype(str)
            .apply(lambda x: "_".join(x), axis=1)
            .astype("category")
        )

    group_idx = group_series.cat.codes.to_numpy()
    n_groups = len(group_series.cat.categories)
    cells_per_group = np.bincount(group_idx, minlength=n_groups)

    if scaling_factor_col is not None:
        if scaling_factor_col not in obs_valid:
            raise KeyError(
                f"scaling factor column {scaling_factor_col!r} not found in `adata_sc.obs`."
            )
        scaling_weights = obs_valid[scaling_factor_col].to_numpy(dtype=np.float32)
    else:
        scaling_weights = np.ones(len(group_idx), dtype=np.float32)

    X_agg = _aggregate_sparse(
        obsm_array=obsm_array,
        group_idx=group_idx,
        cells_per_group=cells_per_group,
        scaling_weights=scaling_weights,
        n_groups=n_groups,
        agg_type=agg_type,
    )

    keep_mask = cells_per_group >= min_cells
    n_dropped = len(keep_mask) - keep_mask.sum()

    if n_dropped > 0:
        logg.info(f"filtered out {n_dropped} grouped categories (< {min_cells} cells)")

    kept_categories = group_series.cat.categories[keep_mask]

    # map original matrix columns to the new aggregated var_names if they exist
    var_names = obsm_raw.columns if hasattr(obsm_raw, "columns") else None
    var_df = pd.DataFrame(index=var_names) if var_names is not None else None

    obs_df = pd.DataFrame(index=kept_categories)
    obs_df["n_cells"] = cells_per_group[keep_mask]

    if len(group_cols) == 1:
        obs_df[group_cols[0]] = kept_categories
    else:
        mapping = obs_valid[group_cols].groupby(group_series, observed=True).first()
        obs_df[group_cols] = mapping.loc[kept_categories]

    adata_agg = AnnData(
        X=X_agg[keep_mask],
        obs=obs_df,
        var=var_df,
    )

    adata_agg.uns["aggregate_params"] = {
        "obsm_key": obsm_key,
        "groupby": groupby,
        "agg_type": agg_type,
        "subset_key": subset_key,
        "scaling_factor_col": scaling_factor_col,
    }

    return adata_agg


def transfer_weights(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    *,
    obsm_key: str = "cell_weights",
    groupby: str = "cell_type",
    batch_key: str | None = None,
    key_added: str = "shears_weights",
    subset_key: str | None = None,
    scaling_factor_col: str | None = None,
    min_cells: int = 20,
    agg_type: str = "sum",
) -> None:
    """\
    Transfer single-cell deconvolution weights strictly into the bulk datasets coordinate space.

    Parameters
    ----------
    adata_sc
        Annotated single-cell dataset containing the learned weights.
    adata_bulk
        Target bulk dataset.
    obsm_key
        Key in `adata_sc.obsm` containing the continuous deconvolution weights.
    groupby
        Column in `adata_sc.obs` defining the cell types or compartments.
    batch_key
        Optional column in `adata_sc.obs` defining biological replicates (e.g., donors or patients). 
        If provided, aggregation applies a hierarchical filter: the `min_cells` threshold is enforced 
        strictly per replicate before collapsing to the final `groupby` categories. This defends 
        against pseudoreplication and ensures signatures are driven by robust biological cohorts.
    key_added
        Key in `adata_bulk.obsm` where the transferred weights will be stored.
    subset_key
        Optional boolean column in `adata_sc.obs` to strictly include certain cells.
    scaling_factor_col
        Optional column in `adata_sc.obs` containing global scaling weights 
        (e.g., mRNA content per cell) to apply during aggregation.
    min_cells
        Minimum number of cells required per group.
    agg_type
        Mathematical aggregation to apply (`'sum'` is required for later compositional fraction calculations).
    """
    group_cols = [batch_key, groupby] if batch_key else [groupby]
    
    logg.info(f"aggregating {obsm_key!r} by {groupby!r} to map to bulk")

    adata_agg = aggregate_obsm(
        adata_sc,
        obsm_key=obsm_key,
        groupby=group_cols,
        subset_key=subset_key,
        min_cells=min_cells,
        agg_type=agg_type,
        scaling_factor_col=scaling_factor_col,
    )

    weights_df = adata_agg.to_df()
    
    if batch_key:
        # adata_agg.obs still contains the separate 'patient_id' and 'cell_type' columns
        # because we passed them as a list to aggregate_obsm. 
        # We group by the target cell_type and sum across the surviving patients.
        weights_df = weights_df.groupby(adata_agg.obs[groupby], observed=True).sum()
    
    # transpose dimensions: (n_cell_types, n_bulk_samples) -> (n_bulk_samples, n_cell_types)
    weights_df = weights_df.T

    # align and strictly map to bulk obs_names
    missing_samples = set(adata_bulk.obs_names) - set(weights_df.index)
    if missing_samples:
        logg.warning(
            f"missing weights for {len(missing_samples)} bulk samples. filling with 0"
        )

    # reindex guarantees a perfect 1:1 dimensional match to adata_bulk.obs
    weights_df = weights_df.reindex(adata_bulk.obs_names, fill_value=0.0)

    adata_bulk.obsm[key_added] = weights_df

    adata_bulk.uns[f"{key_added}_params"] = adata_agg.uns["aggregate_params"]
    adata_bulk.uns[f"{key_added}_params"]["batch_key"] = batch_key

    logg.info(f"transferred weights to `adata_bulk.obsm[{key_added!r}]`")


def compartment_fraction(
    adata_bulk: AnnData,
    *,
    obsm_key: str = "shears_weights",
    compartment_groups: Sequence[str] | str,
    key_added: str = "compartment_fraction",
    inplace: bool = True,
) -> pd.Series | None:
    """\
    Calculate the compositional fraction of a specific biological compartment in bulk samples.

    Reads the aggregated weights from `adata_sc.uns` and calculates the total fraction 
    of the specified groups (e.g., the 'Normal Sink') per bulk sample.

    This function is a critical defense against the "Compositional Trap" (or "Margin Dilution") 
    in downstream differential abundance testing. When studying the Tumor Microenvironment (TME), 
    large variations in normal tissue contamination (e.g., surgical margins) can artificially 
    dilute the fractions of immune and stromal cells. 

    Statistically, raw deconvolution weights should never be normalized or scaled prior to 
    running a Generalized Linear Model (GLM) like `sh.tl.shears_glm`, as altering the matrix 
    destroys linear additivity and violates count-based statistical assumptions, leading to 
    massive false positives.

    Instead, we can calculate the exact fraction of the confounding compartment. 
    By saving this array to `adata_bulk.obs`, it can be passed safely as a covariate in the 
    GLM design matrix (e.g., `design="~ condition + surgical_margin_fraction"`). The GLM 
    will then rigorously regress out the tissue dilution effect without destructively mutating 
    the underlying raw weights.


    Parameters
    ----------
    adata_bulk
        The bulk dataset containing the transferred deconvolution weights in `.obsm`.
    obsm_key
        The key in `adata_bulk.obsm` where the transferred weights are stored.
    compartment_groups
        A string or list of strings indicating the groups that define the macroscopic compartment.
    key_added
        The column name added to `adata_bulk.obs` to store the calculated fractions.
    inplace
        If `True`, adds the series directly to `adata_bulk.obs`.
        If `False`, returns the calculated `pd.Series` directly.
    """
    if obsm_key not in adata_bulk.obsm:
        raise KeyError(
            f"key {obsm_key!r} not found in `adata_bulk.obsm`. "
            "please run `sh.tl.transfer_weights` first."
        )

    if isinstance(compartment_groups, str):
        compartment_groups = [compartment_groups]

    weights_df = pd.DataFrame(adata_bulk.obsm[obsm_key], index=adata_bulk.obs_names)

    # ignore requested groups that dropped out during min_cells filtering
    available_groups = set(weights_df.columns)
    requested_groups = set(compartment_groups)
    missing = requested_groups - available_groups

    if missing:
        logg.warning(
            f"the following groups were missing from `.obsm[{obsm_key!r}]` and will be ignored: {missing}"
        )

    valid_groups = list(requested_groups & available_groups)

    if not valid_groups:
        logg.warning("no valid compartment groups found. returning 0 for all samples")
        fraction_series = pd.Series(0.0, index=adata_bulk.obs_names)
    else:
        target_sum = weights_df[valid_groups].sum(axis=1)
        total_sum = weights_df.sum(axis=1)

        # handle zero division in samples where total sum is 0
        fraction_series = (target_sum / (total_sum + np.finfo(float).eps)).fillna(0.0)

    if inplace:
        adata_bulk.obs[key_added] = fraction_series
        logg.info(
            f"added '{key_added}' (mean: {fraction_series.mean():.2%}) to `adata_bulk.obs`"
        )
        return None

    return fraction_series
