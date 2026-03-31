import logging
import re
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from shears.util._logging import _mute_logger
from shears.util._wrangling import _is_na_strict

from ._filtering import filter_genes_deseq2

logg = logging.getLogger(__name__)


def _build_sum2zero_contrasts(dds: object, groupby: str) -> dict[str, np.ndarray]:
    """\
    Construct sum-to-zero contrast vectors for one-vs-rest comparisons.

    Standard `patsy` design matrices default to treatment coding, which compares 
    everything to a single arbitrary reference level. To find true lineage markers for
    deconvolution, we must compare a target cell type against the average expression
    of all other populations. This function manually weights the target at 1 and
    distributes the negative weights equally across the remaining groups.

    Parameters
    ----------
    dds
        A fitted `pydeseq2.dds.DeseqDataSet` object containing the design matrix.
    groupby
        The categorical column name used to define the groups.

    Returns
    -------
    A dictionary mapping each unique category in `groupby` to its 1D numpy array 
    contrast vector.
    """
    unique_groups = sorted(dds.obs[groupby].dropna().unique())
    n_groups = len(unique_groups)
    design_cols = dds.obsm["design_matrix"].columns

    if not any(col.startswith(f"{groupby}[T.") for col in design_cols):
        raise TypeError(
            f"Could not find categorical dummy variables for {groupby!r} in the design matrix. "
            f"Ensure `adata_pb.obs[{groupby!r}]` is of string or categorical dtype, not numeric."
        )

    dummy_cols = design_cols[design_cols.str.startswith(f"{groupby}[T.")]

    contrasts = {}
    for group in unique_groups:
        contrast_vec = pd.Series(0.0, index=design_cols)
        dummy_name = f"{groupby}[T.{group}]"

        if dummy_name in dummy_cols:
            # patsy uses treatment coding by default unlike r's contr.sum.
            # manually build a one-vs-rest contrast by weighting the target at 1
            # and subtracting the mean of the other n-1 groups.
            contrast_vec.loc[dummy_name] = 1.0

            other_dummies = dummy_cols[dummy_cols != dummy_name]
            contrast_vec.loc[other_dummies] = -1.0 / (n_groups - 1)
        else:
            # Reference level effect is captured by the intercept,
            # contrast against the rest is the negative average of non-reference columns.
            contrast_vec.loc[dummy_cols] = -1.0 / (n_groups - 1)

        contrasts[group] = contrast_vec.values

    return contrasts


def _get_full_rank_mask(
    adata_pb: AnnData,
    design: str,
) -> np.ndarray:
    """\
    Identify genes capable of surviving a generalized linear model without crashing.

    In highly sparse single-cell pseudobulk data, specific genes often lack enough 
    non-zero observations to support the degrees of freedom required by a complex 
    design matrix. This function iterates through the count matrix and uses 
    Cholesky decomposition as a highly optimized, SVD-free algebraic check to 
    preemptively flag rank-deficient genes that would otherwise crash PyDESeq2.

    Parameters
    ----------
    adata_pb
        The pseudobulked AnnData object.
    design
        The full string design formula (e.g., '~dataset + cell_type').

    Returns
    -------
    A 1D boolean numpy array where `True` indicates the gene has full column rank 
    and is safe to pass to the solver.
    """
    try:
        from formulaic import model_matrix  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Using internal model matrices requires `formulaic`. "
            "Please install it with `pip install formulaic`."
        ) from e

    covariates = set(re.findall(r"\w+", design))
    for cov in covariates:
        if cov not in adata_pb.obs:
            raise KeyError(f"Design covariate {cov!r} not found in `adata_pb.obs`.")

    # Missing covariate values cause row-dropping in formulaic, misaligning
    # the count matrix indices during gene-wise filtering.
    is_na_df = pd.DataFrame({col: _is_na_strict(adata_pb.obs[col]) for col in covariates})
    if is_na_df.any().any():
        nan_cols = is_na_df.columns[is_na_df.any()].tolist()
        raise ValueError(
            f"Design covariates contain missing values in columns: {nan_cols}. "
            "Please filter these observations first to maintain valid degrees of freedom."
        )

    obs_clean = adata_pb.obs.copy()

    # Align factor levels with the underlying formulaic model to prevent artificial singularities.
    for col in covariates:
        if isinstance(obs_clean[col].dtype, pd.CategoricalDtype):
            obs_clean[col] = obs_clean[col].cat.remove_unused_categories()

    try:
        design_matrix = model_matrix(design, obs_clean).values
    except Exception as e:
        raise RuntimeError(f"Could not build design matrix: {e}")

    if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
        raise np.linalg.LinAlgError(
            "The base design matrix is already rank-deficient. "
            "Please check your covariates for perfect confounding."
        )

    # Column-major format optimizes sequential gene-wise extraction speeds.
    counts = adata_pb.X.tocsc() if sp.issparse(adata_pb.X) else np.asarray(adata_pb.X)

    n_genes = counts.shape[1]
    n_params = design_matrix.shape[1]
    valid_genes_mask = np.zeros(n_genes, dtype=bool)

    has_expression = counts > 0

    for i in range(n_genes):
        # Materialize a dense vector per gene strictly for boolean indexing speed,
        # avoiding a full dense matrix conversion of the dataset.
        nonzero_obs_mask = (
            has_expression[:, i].toarray().flatten()
            if sp.issparse(counts)
            else has_expression[:, i]
        )

        # A matrix inherently loses full column rank if observations are fewer than parameters.
        if nonzero_obs_mask.sum() < n_params:
            continue

        design_matrix_subset = design_matrix[nonzero_obs_mask, :]
        xtx = design_matrix_subset.T @ design_matrix_subset

        try:
            # Cholesky decomposition strictly requires positive-definite matrices, serving
            # as a highly optimized, SVD-free algebraic check for full column rank.
            np.linalg.cholesky(xtx)
            valid_genes_mask[i] = True
        except np.linalg.LinAlgError:
            pass

    return valid_genes_mask


def _fit_deseq2_model(
    adata_pb: AnnData,
    groupby: str,
    design: str,
    *,
    min_samples: int,
    min_counts: int,
    apply_cholesky_filter: bool,
    n_jobs: int | None = None,
    layer: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """\
    Fit the PyDESeq2 model and compute sum-to-zero Wald tests safely.

    This function orchestrates the internal PyDESeq2 pipeline. It sequentially handles 
    poscounts size-factor estimation, optional algebraic rank filtering, model fitting, 
    and contrast extraction, while actively catching and sequestering expected 
    `LinAlgError` warnings from perfectly confounded subsets.

    Parameters
    ----------
    adata_pb
        The pseudobulked AnnData object.
    groupby
        The grouping variable for contrasts.
    design
        The explicit design formula.
    min_samples
        Minimum samples required to express a gene above `min_counts`.
    min_counts
        Minimum normalized count threshold.
    apply_cholesky_filter
        If `True`, drops genes causing singular matrices prior to fitting.
    n_jobs
        Number of CPUs for multiprocessing.
    layer
        The layer containing raw integer counts.

    Returns
    -------
    wald_tests_per_group
        A dictionary mapping group names to their DESeq2 results DataFrames.
    captured_warnings
        A list of string warnings captured during the solver execution.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as e:
        raise ImportError("pydeseq2 is required for this tool.") from e

    is_quiet = logg.getEffectiveLevel() > logging.DEBUG

    logg.debug("pre-filtering genes using poscounts size factors")
    with _mute_logger(filter_genes_deseq2.__module__, mute=is_quiet):
        keep_mask, _ = filter_genes_deseq2(
            adata_pb,
            min_samples=min_samples,
            min_counts=min_counts,
            layer=layer,
            n_jobs=n_jobs,
            inplace=False,
        )

    keep_genes = keep_mask.copy()

    if apply_cholesky_filter:
        try:
            is_full_rank = _get_full_rank_mask(adata_pb[:, keep_genes], design)

            n_rank_dropped = (~is_full_rank).sum()
            if n_rank_dropped > 0:
                logg.debug(
                    f"dropped {n_rank_dropped} rank-deficient genes via cholesky decomposition "
                    "prior to pydeseq2 initialization"
                )

            keep_genes[keep_genes] = is_full_rank

            if not keep_genes.any():
                raise ValueError("No full-rank genes remain after algebraic filtering.")

        except np.linalg.LinAlgError as e:
            raise e

    captured_warnings = []
    wald_tests_per_group = {}

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        warnings.filterwarnings(
            "error", message="The design matrix is not full rank", category=UserWarning
        )

        logg.debug("initializing and fitting pydeseq2 model")

        adata_fit = adata_pb[:, keep_genes].copy()

        try:
            dds = DeseqDataSet(
                adata=adata_fit, design=design, n_cpus=n_jobs, quiet=is_quiet
            )
        except UserWarning:
            raise np.linalg.LinAlgError(
                f"Design matrix for {design!r} is not full rank. Check for perfectly confounded covariates."
            ) from None

        dds.fit_size_factors(fit_type="poscounts")
        dds.deseq2()

        contrasts = _build_sum2zero_contrasts(dds, groupby=groupby)

        logg.debug(f"computing wald tests for {len(contrasts)} groups")
        for group, contrast_vec in contrasts.items():
            ds = DeseqStats(dds, contrast=contrast_vec, quiet=is_quiet, n_cpus=n_jobs)
            ds.summary()
            wald_test_results = ds.results_df.reset_index(names="var_names")
            wald_tests_per_group[group] = wald_test_results

        for warning in caught_warnings:
            msg = str(warning.message)
            if "The design matrix is not full rank" not in msg:
                captured_warnings.append(msg)

    return wald_tests_per_group, captured_warnings
