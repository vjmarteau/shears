import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from shears.util._wrangling import _get_clean_mask
from shears.util._parallel import _parallelize_with_joblib

logg = logging.getLogger(__name__)


def _calculate_stratified_downsample(
    obs_df: pd.DataFrame,
    groupby: str,
    batch_key: str,
    min_cells: int,
    global_floor: int,
    batch_cap: int,
    random_state: int,
) -> np.ndarray:
    """Core mathematical engine for stratified downsampling."""
    keep_mask = pd.Series(False, index=obs_df.index)

    if not isinstance(obs_df[groupby].dtype, pd.CategoricalDtype):
        obs_df[groupby] = obs_df[groupby].astype("category")
    obs_df[groupby] = obs_df[groupby].cat.remove_unused_categories()

    original_groups = set(obs_df[groupby].unique())

    obs_df["n_per_batch"] = obs_df.groupby([groupby, batch_key], observed=True)[
        groupby
    ].transform("size")

    # Ridge regression penalization requires stable intra-batch variance.
    # If a patient only has a handful of cells for a given state, those cells lack
    # the statistical degrees of freedom to form a reliable biological profile.
    # Including them risks overfitting the Ridge solver to technical noise,
    # transcriptomic dropouts, or stray mis-annotated cells.
    is_noise_replicate = obs_df["n_per_batch"] < min_cells

    if is_noise_replicate.any():
        n_noise = is_noise_replicate.sum()
        logg.info(
            f"dropped {n_noise:,} cells from minor replicates with fewer than {min_cells} cells"
        )
        obs_df = obs_df[~is_noise_replicate].copy()

        surviving_groups = set(obs_df[groupby].unique())
        wiped_out = original_groups - surviving_groups
        if wiped_out:
            logg.warning(
                f"Strict `min_cells={min_cells}` filtering completely eliminated rare populations: {wiped_out}. "
                "Consider lowering the threshold to preserve biological diversity."
            )

    group_sizes = obs_df[groupby].value_counts()
    rare_groups = set(group_sizes[group_sizes < global_floor].index)
    abundant_groups = set(group_sizes[group_sizes >= global_floor].index)

    logg.info(
        f"identified {len(rare_groups)} rare and {len(abundant_groups)} abundant groups using a floor of {global_floor}"
    )

    # Exempt rare populations from the batch cap so the downstream
    # ridge solver does not ignore minority signals during loss minimization.
    is_rare_population = obs_df[groupby].isin(rare_groups)
    rare_indices = obs_df[is_rare_population].index
    keep_mask.loc[rare_indices] = True

    df_abundant = obs_df[~is_rare_population]
    if not df_abundant.empty:
        # random sampling limits redundant biological clones from dominating
        # the global L2 regularization loss function.
        sampled_abundant = (
            df_abundant.sample(frac=1.0, random_state=random_state)
            .groupby([groupby, batch_key], observed=True)
            .head(batch_cap)
        )
        keep_mask.loc[sampled_abundant.index] = True

    return keep_mask.to_numpy()


def downsample_reference(
    adata_sc: AnnData,
    *,
    groupby: str,
    batch_key: str,
    min_cells: int = 20,
    global_floor: int = 5000,
    batch_cap: int = 250,
    subset_key: str | None = None,
    key_added: str = "shears_downsampled",
    random_state: int = 0,
    inplace: bool = True,
    subset: bool = False,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """\
    Downsample a single-cell reference using a hybrid stratified strategy.

    To optimize non-negative Ridge regression, the reference matrix must capture true 
    intra-sample single-cell heterogeneity without allowing massive class imbalances to 
    dominate the global L2 regularization penalty. This function performs dynamic range 
    compression by dropping noisy minor replicates that lack the statistical degrees of 
    freedom to establish stable variance, preserving globally rare populations entirely, 
    and capping hyper-abundant cell types per batch. This discards redundant biological 
    clones and prevents abundant states from masking rare signals during deconvolution.

    Parameters
    ----------
    adata_sc
        The annotated data matrix of the single-cell reference.
    groupby
        Key in `adata_sc.obs` containing the target cell type or state annotations.
    batch_key
        Key in `adata_sc.obs` containing technical batch or biological replicate 
        information (e.g., patient ID or sample ID).
    min_cells
        The strict lower limit of cells required for a specific cell type within a 
        single batch. Replicates falling below this lack the degrees of freedom to 
        establish a reliable intra-batch biological profile and are excluded as noise.
    global_floor
        The absolute minimum number of cells required globally across all batches 
        for a cell type to be considered "abundant". Rare populations falling below 
        this floor are fully preserved and exempt from `batch_cap` downsampling to 
        prevent the GLM solver from ignoring them.
    batch_cap
        The maximum number of cells to retain per cell type, per batch. Hyper-abundant 
        populations are randomly downsampled to this ceiling to remove redundant 
        biological clones and prevent collinearity imbalances. Biologically, the first 
        ~50 cells establish the core mean (centroid), while cells 51-200 establish 
        true intra-sample variance (e.g., natural activation gradients, dropout rates). 
        Beyond ~200 cells, sampling yields diminishing returns as additional cells 
        act as mathematically redundant clones.
    subset_key
        Optional boolean column in `adata_sc.obs` to pre-filter the dataset (e.g., 
        `'passed_qc'`). If provided, only cells where this key is True are considered.
    key_added
        The key added to `adata_sc.obs` (boolean mask) and `adata_sc.uns` (parameters 
        dictionary) if `inplace=True` and `subset=False`.
    random_state
        Seed for the random number generator used during abundant population sampling.
    inplace
        If True, mutate the `adata_sc` object. If False, bypass AnnData mutation and 
        return the boolean mask and parameters dictionary directly.
    subset
        If True and `inplace=True`, destructively subset the `adata_sc` object in 
        memory, discarding filtered cells. If False, non-destructively append a 
        boolean mask to `adata_sc.obs[key_added]`.

    Returns
    -------
    If `inplace=False`, returns a tuple containing:
        - `is_downsampled_keep`: A boolean numpy array masking the retained cells.
        - `params_dict`: A dictionary containing the threshold parameters used.
    Otherwise, returns `None` and updates `adata_sc` based on `inplace` and `subset`.
    """
    if groupby not in adata_sc.obs:
        raise KeyError(f"groupby column {groupby!r} not found in `adata_sc.obs`.")
    if batch_key not in adata_sc.obs:
        raise KeyError(f"batch_key column {batch_key!r} not found in `adata_sc.obs`.")

    is_valid_cell = _get_clean_mask(
        adata_sc, obs_columns=[groupby, batch_key]
    ).to_numpy()

    if subset_key is not None:
        if subset_key not in adata_sc.obs:
            raise KeyError(f"subset_key {subset_key!r} not found in `adata_sc.obs`.")
        is_valid_cell &= adata_sc.obs[subset_key].astype(bool).to_numpy()

    obs_valid = adata_sc.obs.iloc[is_valid_cell][[groupby, batch_key]].copy()

    keep_mask_valid = _calculate_stratified_downsample(
        obs_df=obs_valid,
        groupby=groupby,
        batch_key=batch_key,
        min_cells=min_cells,
        global_floor=global_floor,
        batch_cap=batch_cap,
        random_state=random_state,
    )

    # Map the valid subset mask back to the global AnnData dimensions
    is_downsampled_keep = np.zeros(adata_sc.n_obs, dtype=bool)
    is_downsampled_keep[is_valid_cell] = keep_mask_valid

    n_original = is_valid_cell.sum()
    n_kept = is_downsampled_keep.sum()
    logg.info(
        f"downsampled reference from {n_original:,} to {n_kept:,} cells capping at {batch_cap} per batch"
    )

    params_dict = {
        "groupby": groupby,
        "batch_key": batch_key,
        "min_cells": min_cells,
        "global_floor": global_floor,
        "batch_cap": batch_cap,
        "subset_key": subset_key,
        "random_state": random_state,
        "n_cells_retained": int(n_kept),
    }

    if not inplace:
        return is_downsampled_keep, params_dict

    if subset:
        logg.info("subsetting anndata inplace")
        adata_sc._inplace_subset_obs(is_downsampled_keep)
        adata_sc.uns[key_added] = {"params": params_dict}
    else:
        logg.info(f"adding `{key_added}` to obs")
        adata_sc.obs[key_added] = is_downsampled_keep
        adata_sc.uns[key_added] = {"params": params_dict}

    return None


def _compute_ridge_weights_parallel(
    bulk_mat: np.ndarray,
    sc_mat: np.ndarray | sparse.csc_matrix,
    alpha_val: float,
    random_state: int,
    n_jobs: int | None,
    backend: str,
) -> np.ndarray:
    """Distribute non-negative Ridge regression across bulk samples."""
    from joblib import delayed


    def _deconvolute(bulk_sample: np.ndarray) -> np.ndarray:
        """Fit a positive ridge regression model for a single bulk sample."""
        from sklearn.linear_model import Ridge
        from threadpoolctl import threadpool_limits

        # Lock underlying C-level threads in parallel workers to prevent thread oversubscription.
        with threadpool_limits(limits=1):
            model = Ridge(alpha=alpha_val, positive=True, random_state=random_state)
            fit = model.fit(sc_mat, bulk_sample)
        return fit.coef_

    jobs = (delayed(_deconvolute)(bulk_mat[i, :]) for i in range(bulk_mat.shape[0]))

    weights_list = list(
        _parallelize_with_joblib(
            jobs, total=bulk_mat.shape[0], n_jobs=n_jobs, backend=backend
        )
    )
    return np.array(weights_list).T


def cell_weights(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    *,
    subset_key: str | None = None,
    alpha_callback: Callable[[AnnData], float] = lambda ad: float(ad.shape[0]),
    layer_sc: str | None = "ref_scaled",
    layer_bulk: str | None = "ref_scaled",
    key_added: str = "cell_weights",
    random_state: int = 0,
    n_jobs: int | None = None,
    backend: str = "loky",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """\
    Compute a bulk-by-cell weight matrix using non-negative ridge regression.

    Single-cell reference matrices are inherently multicollinear (e.g., CD8+ T-cells 
    and CD4+ T-cells share massive overlapping gene modules). Without L2 regularization 
    (`alpha_callback`), the model matrix would become ill-conditioned, causing the 
    solver to wildly oscillate weights between highly similar lineages. The positivity 
    constraint is biologically non-negotiable, as a tissue cannot contain negative 
    cell fractions.

    To formulate the deconvolution GLM, the algorithm transposes the single-cell 
    matrix so that genes become the observations (samples) and cells become the 
    features (predictors) for the solver.

    Parameters
    ----------
    adata_sc
        The annotated single-cell reference matrix.
    adata_bulk
        The annotated bulk RNA-seq mixture matrix.
    subset_key
        Boolean column in `adata_sc.obs` to filter the reference before solving.
    alpha_callback
        A function returning the L2 regularization penalty based on the reference shape.
    layer_sc
        Layer in `adata_sc` containing the scaled, log-normalized counts.
    layer_bulk
        Layer in `adata_bulk` containing the scaled, log-normalized counts.
    key_added
        Key added to `adata_sc.obsm` (weight matrix) and `adata_sc.uns` (parameters).
    random_state
        Seed for the random number generator in the solver.
    n_jobs
        Number of parallel jobs to run across bulk samples.
    backend
        Joblib backend to use (defaults to "loky").
    inplace
        If True, mutates `adata_sc` in place. If False, returns the DataFrame.
    """
    if not np.array_equal(adata_sc.var_names, adata_bulk.var_names):
        raise ValueError(
            "`var_names` in `adata_sc` and `adata_bulk` must be perfectly aligned."
        )
    if layer_bulk is not None and layer_bulk not in adata_bulk.layers:
        raise KeyError(f"Layer {layer_bulk!r} not found in `adata_bulk.layers`.")
    if layer_sc is not None and layer_sc not in adata_sc.layers:
        raise KeyError(f"Layer {layer_sc!r} not found in `adata_sc.layers`.")

    solver_inclusion_mask = None
    adata_solver = adata_sc

    if subset_key is not None:
        if subset_key not in adata_sc.obs:
            raise KeyError(f"subset_key {subset_key!r} not found in `adata_sc.obs`.")

        solver_inclusion_mask = adata_sc.obs[subset_key].to_numpy(dtype=bool)
        adata_solver = adata_sc[solver_inclusion_mask]

        logg.info(
            f"subsetting reference to {solver_inclusion_mask.sum():,} / {adata_sc.n_obs:,} cells"
        )

        if subset_key in adata_sc.uns and "params" in adata_sc.uns[subset_key]:
            groupby = adata_sc.uns[subset_key]["params"].get("groupby")
            if groupby and groupby in adata_sc.obs:
                counts = adata_solver.obs[groupby].value_counts()
                if not counts.empty:
                    ratio = counts.max() / counts.min()
                    logg.info(f"soft prior ratio (max:min) is {ratio:.1f}:1")

    alpha_val = alpha_callback(adata_solver)

    if key_added in adata_sc.uns and key_added in adata_sc.obsm:
        cached_params = adata_sc.uns[key_added].get("params", {})
        if (
            cached_params.get("layer_sc") == layer_sc
            and cached_params.get("layer_bulk") == layer_bulk
            and cached_params.get("alpha_val") == alpha_val
            and cached_params.get("subset_key") == subset_key
        ):
            logg.info(
                f"found cached weights in `.obsm[{key_added!r}]`. skipping computation"
            )
            # Cache consistency fix
            if not inplace:
                return adata_sc.obsm[key_added].copy()
            return None

    logg.info("calculating cell weights via positive ridge regression")

    bulk_x = adata_bulk.layers[layer_bulk] if layer_bulk else adata_bulk.X
    if sparse.issparse(bulk_x):
        bulk_x = bulk_x.toarray()

    # Ensure bulk_mat is in standard C-contiguous layout for the C-solver.
    bulk_target_mat = np.ascontiguousarray(bulk_x, dtype=np.float64)

    sc_x = adata_solver.layers[layer_sc] if layer_sc else adata_solver.X
    if sparse.issparse(sc_x):
        # A transposed CSR matrix natively becomes a CSC matrix, which is highly efficient
        # for scikit-learn's column-wise operations in linear models.
        sc_design_mat = sc_x.tocsc().T
    else:
        # Transposing a C-contiguous array creates an F-contiguous view without moving bytes.
        # If passed to scikit-learn C-backend, it will silently copy the matrix inside
        # every parallel worker to realign it, resulting in RAM spikes. We physically realign it
        # here once to prevent worker duplication.
        sc_design_mat = np.ascontiguousarray(sc_x.T, dtype=np.float64)

    solved_weights = _compute_ridge_weights_parallel(
        bulk_target_mat, sc_design_mat, alpha_val, random_state, n_jobs, backend
    )

    cell_weights_df = pd.DataFrame(
        0.0, index=adata_sc.obs_names, columns=adata_bulk.obs_names, dtype=np.float64
    )

    if solver_inclusion_mask is not None:
        solved_df = pd.DataFrame(
            solved_weights,
            index=adata_solver.obs_names,
            columns=adata_bulk.obs_names,
            dtype=np.float64,
        )
        cell_weights_df.update(solved_df)
    else:
        cell_weights_df.loc[:, :] = solved_weights

    if inplace:
        adata_sc.obsm[key_added] = cell_weights_df
        adata_sc.uns[key_added] = {
            "params": {
                "layer_sc": layer_sc,
                "layer_bulk": layer_bulk,
                "random_state": random_state,
                "alpha_val": alpha_val,
                "subset_key": subset_key,
            }
        }
        logg.info(f"added {key_added!r} to adata.obsm")
        return None

    return cell_weights_df


def calculate_scaling_factors(
    adata_sc: AnnData,
    *,
    groupby: str,
    batch_key: str | None = None,
    weight_col: str = "n_genes",
    min_cells_per_batch: int = 20,
    clip_fraction: float = 0.05,
    key_added: str = "mRNA_scaling_factor",
    inplace: bool = True,
) -> pd.Series | None:
    """\
    Calculate intrinsic mRNA scaling factors to correct for biological yield bias.

    Unlike naive composition-based scaling, this function isolates the pure biological 
    transcriptional depth of each cell state, stabilizing it against the global median. 
    This prevents mathematical inflation of rare cells (or downsampled lineages) during 
    downstream GLM deconvolution.

    Parameters
    ----------
    adata_sc
        The annotated single-cell reference matrix.
    groupby
        Column in `adata_sc.obs` containing cell-type or state annotations.
    batch_key
        Optional column in `adata_sc.obs` for batch-aware baseline calculation.
        Highly recommended to prevent heavily sequenced batches from skewing the median.
    weight_col
        Column in `adata_sc.obs` holding the raw mRNA proxy (e.g., total counts or n_genes).
    min_cells_per_batch
        Minimum number of valid cells required in a group/batch to calculate a stable median. 
        Groups below this threshold receive a factor of `NaN` to prevent ambient/doublet noise.
    clip_fraction
        Fraction of extreme scaling factors to clip (winsorize) at the top and bottom tails.
    key_added
        Column name under which to save the final scaling factors in `adata_sc.obs`.
    inplace
        If `True`, inserts the factors into `adata_sc.obs` and updates the `.uns` state machine.
        If `False`, returns the scaling factors as a `pandas.Series`.

    Returns
    -------
    Depending on `inplace`, either updates the `AnnData` object or returns a `pandas.Series`.
    """
    if weight_col not in adata_sc.obs.columns:
        raise KeyError(
            f"column {weight_col!r} missing from `adata_sc.obs`. "
            f"Compute it first, e.g., via `sc.pp.filter_cells()`."
        )

    logg.info("calculating pure mrna scaling factors")

    obs_columns = [groupby] if batch_key is None else [batch_key, groupby]
    clean_mask = _get_clean_mask(adata_sc, obs_columns)
    
    df_clean = adata_sc.obs.loc[clean_mask, obs_columns + [weight_col]]

    grouped = df_clean.groupby(obs_columns, observed=True)
    group_sizes = grouped.size()
    
    valid_groups = group_sizes[group_sizes >= min_cells_per_batch].index
    
    if len(valid_groups) == 0:
        raise ValueError(
            f"no groups passed the `min_cells_per_batch={min_cells_per_batch}` threshold. "
            "Check your clustering or lower the threshold."
        )
    
    # Pure mRNA Scaling
    # We only compute the median for groups that survived the noise filter
    stable_baselines = grouped[weight_col].median().loc[valid_groups]
    
    # The global median is calculated strictly from the clean, unfiltered cells 
    # to provide a stable, dataset wide denominator (centers factors around 1.0)
    global_median = df_clean[weight_col].median()
    
    raw_factors = stable_baselines / global_median

    # We map the group-level factors back to individual cells. 
    # Cells in noisy micro-clusters or with NaN metadata will safely receive NaN.
    if batch_key is None:
        scaling_factors = adata_sc.obs[groupby].map(raw_factors)
    else:
        mapped_idx = pd.MultiIndex.from_arrays([adata_sc.obs[batch_key], adata_sc.obs[groupby]])
        scaling_factors = pd.Series(mapped_idx.map(raw_factors), index=adata_sc.obs_names)

    # Outlier mitigation (winsorizing)
    if clip_fraction > 0.0:
        lower_bound = float(scaling_factors.quantile(clip_fraction))
        upper_bound = float(scaling_factors.quantile(1.0 - clip_fraction))
        scaling_factors = scaling_factors.clip(lower=lower_bound, upper=upper_bound)
        logg.info(
            f"clipped extreme scaling factors beyond the {clip_fraction:.0%} "
            f"and {1-clip_fraction:.0%} quantiles"
        )

    if inplace:
        adata_sc.obs[key_added] = np.asarray(scaling_factors)
        
        adata_sc.uns[f"{key_added}_params"] = {
            "groupby": groupby,
            "batch_key": batch_key,
            "weight_col": weight_col,
            "min_cells_per_batch": min_cells_per_batch,
            "clip_fraction": clip_fraction,
        }
        logg.info(f"added normalized mRNA weights to `.obs[{key_added!r}]` and `.uns`")
        return None
    
    return scaling_factors
