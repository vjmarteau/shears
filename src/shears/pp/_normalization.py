import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from shears.util._wrangling import _get_clean_mask
from sklearn.preprocessing import StandardScaler, quantile_transform

logg = logging.getLogger(__name__)


def quantile_norm(adata: AnnData, *, layer=None, key_added="quantile_norm", **kwargs):
    """Perform quantile normalization on AnnData object

    Stores the normalized data in a new layer with the key `key_added`.
    """
    X = adata.X if layer is None else adata.layers[layer]
    adata.layers[key_added] = quantile_transform(X, **kwargs)


def scale_to_reference(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    *,
    layer: str | None = None,
    key_added: str = "ref_scaled",
    copy: bool = False,
) -> tuple[AnnData, AnnData] | None:
    """\
    Scale bulk query data using the variance learned from the single-cell reference.
    """
    if copy:
        adata_sc = adata_sc.copy()
        adata_bulk = adata_bulk.copy()

    if not adata_sc.var_names.equals(adata_bulk.var_names):
        raise ValueError(
            "the var_names of adata_sc and adata_bulk must be identical and "
            "perfectly aligned before scaling"
        )

    mat_sc = adata_sc.X if layer is None else adata_sc.layers[layer]
    mat_bulk = adata_bulk.X if layer is None else adata_bulk.layers[layer]

    if sp.issparse(mat_sc) and not sp.isspmatrix_csr(mat_sc):
        mat_sc = mat_sc.tocsr()
    if sp.issparse(mat_bulk) and not sp.isspmatrix_csr(mat_bulk):
        mat_bulk = mat_bulk.tocsr()

    adata_sc.layers[key_added] = mat_sc.copy()
    adata_bulk.layers[key_added] = mat_bulk.copy()

    if "shears_ref_std" in adata_sc.var:
        logg.info("using cached variance from .var['shears_ref_std']")
        ref_std = adata_sc.var["shears_ref_std"].values

        # instantiate a blank scaler
        scaler = StandardScaler(with_mean=False, copy=False)

        # inject learned attributes
        scaler.scale_ = ref_std
        scaler.var_ = ref_std ** 2
        scaler.mean_ = None

        scaler.transform(adata_sc.layers[key_added])
        scaler.transform(adata_bulk.layers[key_added])

    else:
        logg.info("scaling features to reference variance")
        scaler = StandardScaler(with_mean=False, copy=False)
        scaler.fit_transform(adata_sc.layers[key_added])
        scaler.transform(adata_bulk.layers[key_added])

        adata_sc.var["shears_ref_std"] = scaler.scale_

    return (adata_sc, adata_bulk) if copy else None


def calculate_batch_diversity(
    adata_sc: AnnData,
    *,
    groupby: str,
    batch_key: str,
    replicate_key: str | None = None,
    min_cells: int = 20,
    min_diversity: int = 5,
    key_added: str = "is_diverse_batch",
    inplace: bool = True,
) -> pd.Series | None:
    """\
    Flag robust sequencing batches based on intra-replicate cell type diversity.

    Plate-based assays (e.g., SMARTer, scTrio-seq) or sorted batches often contain 
    extreme biological biases, capturing only a single lineage like epithelial cells. 
    This function calculates a diversity score by counting how many distinct cell 
    types possess sufficient cell counts within the batch, and flags platforms that 
    fail to meet the diversity threshold.

    Parameters
    ----------
    adata_sc
        The annotated single-cell reference matrix.
    groupby
        Column containing cell type annotations.
    batch_key
        Column containing the sequencing platform, batch, or study ID.
    replicate_key
        Optional column containing the biological replicate (e.g., patient ID). 
        If provided, a cell type must have `min_cells` within a *single 
        replicate* to count toward the platform's overall diversity score.
    min_cells
        Minimum number of cells a cell type must have to be considered "valid".
    min_diversity
        Minimum number of valid cell types a batch must contain to be flagged as robust.
    key_added
        Key added to `.obs` containing the boolean mask of cells in robust batches.
    inplace
        If True, adds the mask to `.obs`. If False, returns the pd.Series.
    """
    logg.info(f"evaluating `{batch_key}` diversity to flag biased platforms")
    
    obs_columns = [batch_key, groupby]
    if replicate_key:
        obs_columns.append(replicate_key)

    is_valid_cell = _get_clean_mask(adata_sc, obs_columns)
    df_valid = adata_sc.obs.loc[is_valid_cell, obs_columns]
    
    grouping_keys = [batch_key]
    if replicate_key:
        grouping_keys.append(replicate_key)
    grouping_keys.append(groupby)

    robust_diversity = (
        df_valid
        .groupby(grouping_keys, observed=True)
        .size()
        .loc[lambda x: x >= min_cells]
        .reset_index(name="valid_cells")
        .groupby(batch_key, observed=True)[groupby]
        .nunique()
    )
    
    # Ensure categories that were completely wiped out are still represented as 0
    all_categories = adata_sc.obs[batch_key].cat.categories
    robust_diversity = robust_diversity.reindex(all_categories, fill_value=0)
    
    valid_batches = robust_diversity[robust_diversity >= min_diversity].index
    dropped_batches = robust_diversity[robust_diversity < min_diversity].index
    
    if not dropped_batches.empty:
        logg.warning(
            f"Flagged {len(dropped_batches)} batches as biased or low-diversity "
            f"(< {min_diversity} cell types): {list(dropped_batches)}."
        )
        
    is_diverse = adata_sc.obs[batch_key].isin(valid_batches) & is_valid_cell

    if inplace:
        adata_sc.obs[key_added] = is_diverse
        logg.info(f"added boolean mask to `.obs[{key_added!r}]`")
        return None
    
    return is_diverse


def _calculate_anchor_cvs(
    metadata_df: pd.DataFrame, 
    groupby: str, 
    var_keys: list[str],
    counts_key: str,
    min_cells: int,
) -> pd.Series:
    """\
    Calculate the Coefficient of Variation (CV) of median mRNA yield across biological replicates.
    
    A lower CV indicates a cell type whose total structural RNA content is highly 
    stable across different patients or sequencing platforms, making it an ideal 
    denominator for compositional scaling.
    """
    grouping_cols = var_keys + [groupby]
    
    group_counts = metadata_df.groupby(grouping_cols, observed=True).size()
    valid_replicates = group_counts[group_counts >= min_cells].index
    
    replicate_medians = metadata_df.groupby(grouping_cols, observed=True)[counts_key].median()
    valid_medians = replicate_medians.loc[valid_replicates]
    
    group_stats = valid_medians.groupby(groupby, observed=True).agg(["mean", "std"])
    cv = group_stats["std"] / group_stats["mean"]
    
    return cv.dropna()


def calculate_scaling_factors(
    adata_sc: AnnData,
    *,
    groupby: str,
    batch_key: str | None = None,
    replicate_key: str | None = None,
    subset_key: str | None = None,
    reference_group: str | Literal["automatic"] | None = "automatic",
    exclude_groups: Sequence[str] | None = None,
    counts_key: str = "total_counts",
    min_cells: int = 20,
    clip_fraction: float = 0.05,
    key_added: str = "mRNA_scaling_factor",
    inplace: bool = True,
) -> pd.Series | None:
    """\
    Calculate intrinsic mRNA scaling factors to correct for biological yield bias.

    Corrects for the biological reality that large cells (e.g., macrophages) contain 
    more total mRNA than small cells (e.g., T-cells). 
    
    Inspired by compositional data analysis (scCODA), this function uses a 
    `reference_group` to establish a stable baseline. This is highly recommended 
    for atlases with mixed sorting strategies (e.g., CD45+ vs. whole tissue), 
    as it prevents the absence of large structural cells from skewing the denominator.

    Parameters
    ----------
    adata_sc
        An annotated data matrix of single cells.
    groupby
        The column in `adata_sc.obs` containing fine-grained cell type annotations.
    batch_key
        The column in `adata_sc.obs` containing batch or sequencing platform labels.
    replicate_key
        The column defining independent biological replicates (e.g., patient or donor id). 
        Used to calculate the cross-replicate coefficient of variation (CV) to punish 
        highly variable states (like cancer cells) during automatic anchor selection.
    subset_key
        Column in `adata_sc.obs` containing a boolean mask. If provided, the 
        median baseline is calculated strictly from this subset of high-quality cells, 
        protecting the biological anchor from being skewed by noisy or minor replicates. 
        The resulting scaling factors are still projected back to the entire dataset. 
    reference_group
        The cell type to use as the baseline (factor = 1.0). If `"automatic"`, 
        selects the cell type with the most stable mRNA yield across all batches/replicates. 
        If `None`, falls back to the batch global median (not recommended for sorted data).
    exclude_groups
        List of cell types to explicitly blacklist from automatic anchor selection.
    counts_key
        The column in `adata_sc.obs` containing total mRNA yield per cell.
    min_cells
        Minimum cells required per group (within a batch) to calculate a stable median.
    clip_fraction
        Fraction of extreme outliers to clip at the top and bottom of the final distribution.
    key_added
        The key in `.obs` and `.uns` where results are saved.
    inplace
        If `True`, mutates `adata_sc` in place. If `False`, returns the pd.Series.
    """
    if counts_key not in adata_sc.obs.columns:
        raise KeyError(f"Column {counts_key!r} missing from `adata_sc.obs`.")

    max_counts = adata_sc.obs[counts_key].max()
    if max_counts < 50:
        logg.warning(
            f"The maximum value of {counts_key!r} is unusually low at {max_counts:.2f}. "
            "Scaling factors should be calculated on raw, unnormalized counts. "
            "Please verify that this column does not contain log1p-transformed data."
        )

    logg.info("calculating biological mRNA scaling factors")

    internal_batch_key = batch_key if batch_key else "_dummy_batch"

    obs_columns = [groupby]
    if batch_key:
        obs_columns.append(batch_key)
    if replicate_key:
        obs_columns.append(replicate_key)

    is_valid_cell = _get_clean_mask(adata_sc, obs_columns)

    if subset_key is not None:
        subset_mask = adata_sc.obs[subset_key].fillna(False).astype(bool)
        logg.info(
            f"restricting baseline calculation to {subset_mask.sum():,} cells defined by {subset_key!r}"
        )
        is_valid_cell &= subset_mask

    metadata_df = adata_sc.obs.loc[is_valid_cell, obs_columns + [counts_key]].copy()

    # Unify batch processing by injecting a global key if no technical batch is provided
    if not batch_key:
        metadata_df[internal_batch_key] = "all_cells"

    for col in [groupby, internal_batch_key]:
        if not isinstance(metadata_df[col].dtype, pd.CategoricalDtype):
            metadata_df[col] = metadata_df[col].astype("category")

    grouped = metadata_df.groupby([internal_batch_key, groupby], observed=True)
    group_sizes = grouped.size()

    valid_groups = group_sizes[group_sizes >= min_cells].index
    if len(valid_groups) == 0:
        raise ValueError(f"No groups passed the `min_cells={min_cells}` threshold.")

    yield_baselines = grouped[counts_key].median().loc[valid_groups]

    # The cell type anchor must be ubiquitous across all platforms.
    # a missing anchor in a single batch would evaluate to NaN, corrupting the scaling matrix.
    n_batches = metadata_df[internal_batch_key].nunique()
    valid_batch_counts = (
        valid_groups.to_frame(index=False).groupby(groupby, observed=True).size()
    )
    ubiquitous_groups = valid_batch_counts[valid_batch_counts == n_batches].index

    # Pre calculate CVs for both automatic search and manual logging
    var_keys = [internal_batch_key, replicate_key] if replicate_key else [internal_batch_key]
    cv_series = _calculate_anchor_cvs(metadata_df, groupby, var_keys, counts_key, min_cells)

    if reference_group == "automatic":
        if not batch_key and not replicate_key:
            logg.warning(
                "Cannot use 'automatic' reference without a batch or replicate key. Falling back to global."
            )
            reference_group = None
        else:
            # Intersect with ubiquitous groups so we only evaluate mathematically viable anchors
            valid_candidates = cv_series.loc[
                cv_series.index.intersection(ubiquitous_groups)
            ]

            if exclude_groups:
                valid_candidates = valid_candidates.drop(
                    index=exclude_groups, errors="ignore"
                )

            if valid_candidates.empty:
                logg.warning(
                    "No cell type is universally shared across all batches. Falling back to global median."
                )
                reference_group = None
            else:
                top_3 = valid_candidates.nsmallest(3)
                reference_group = top_3.index[0]
                min_cv = top_3.iloc[0]
                runner_ups = ", ".join([f"{k} (cv: {v:.2f})" for k, v in top_3.items()])

                logg.info(
                    f"automatically anchored to {reference_group!r} with the lowest coefficient of variation at {min_cv:.2f}"
                )
                logg.info(f"top reference candidates by cv: {runner_ups}")

    elif reference_group is not None:
        if reference_group in cv_series.index:
            ref_cv = cv_series.loc[reference_group]
            logg.info(
                f"anchored to manual reference {reference_group!r} (cv: {ref_cv:.2f})"
            )
        else:
            logg.info(f"anchored to manual reference {reference_group!r}")

    if reference_group is not None:
        if reference_group not in yield_baselines.index.get_level_values(groupby):
            raise ValueError(
                f"Reference group {reference_group!r} not found in any valid batch."
            )

        # Isolate technical capture efficiency from biological yield by normalizing
        # each cell type strictly against the anchors yield within that specific batch.
        anchor_per_batch = yield_baselines.xs(reference_group, level=groupby)

        all_batches = metadata_df[internal_batch_key].dropna().unique()
        missing_batches = set(all_batches) - set(anchor_per_batch.index)

        if missing_batches:
            raise ValueError(
                f"The requested reference group {reference_group!r} is missing from batches: {list(missing_batches)}. "
                "Either choose a more ubiquitous cell type, or use a `subset_key` to exclude these batches first."
            )

        scaling_ratios = yield_baselines.div(anchor_per_batch, level=internal_batch_key)
    else:
        anchor_per_batch = metadata_df.groupby(internal_batch_key, observed=True)[
            counts_key
        ].median()
        scaling_ratios = yield_baselines.div(anchor_per_batch, level=internal_batch_key)

    target_keys = pd.MultiIndex.from_arrays(
        [
            (
                adata_sc.obs[batch_key]
                if batch_key
                else pd.Series("all_cells", index=adata_sc.obs_names)
            ),
            adata_sc.obs[groupby],
        ]
    )
    has_ratio_mask = target_keys.isin(scaling_ratios.index)

    scaling_factors = pd.Series(
        np.where(has_ratio_mask, target_keys.map(scaling_ratios), np.nan),
        index=adata_sc.obs_names,
        dtype=np.float64,
    )

    # Winsorize extreme biological outliers to stabilize the scaling matrix
    if clip_fraction > 0.0:
        lower_bound = float(scaling_factors.quantile(clip_fraction))
        upper_bound = float(scaling_factors.quantile(1.0 - clip_fraction))
        scaling_factors = scaling_factors.clip(lower=lower_bound, upper=upper_bound)

    if inplace:
        adata_sc.obs[key_added] = np.asarray(scaling_factors)
        adata_sc.uns[key_added] = {
            "params": {"groupby": groupby, "reference_group": reference_group}
        }
        logg.info(f"added {key_added!r} to adata.obs")
        return None

    return scaling_factors
