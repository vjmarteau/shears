import logging
import re
import time
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from shears.util._wrangling import _is_na_strict
from tqdm.auto import tqdm

from ._utils import _fit_deseq2_model

logg = logging.getLogger(__name__)


def rank_genes_sum2zero(
    adata_pb: AnnData,
    groupby: str,
    *,
    layer: str | None = None,
    subset_key: str | None = None,
    design: str | None = None,
    min_samples: int = 3,
    min_counts: int = 5,
    rank_filter_method: Literal["cholesky"] | None = None,
    gene_symbols: str | None = None,
    key_added: str = "rank_genes_sum2zero",
    n_jobs: int | None = None,
    inplace: bool = True,
) -> dict | None:
    """\
    Rank genes using PyDESeq2 with sum-to-zero (one-vs-rest) contrasts.

    In single-cell deconvolution, isolating highly specific lineage markers is critical. 
    Standard differential expression often highlights "pan-markers" shared across multiple 
    lineages. By utilizing sum-to-zero contrasts, this function actively penalizes genes 
    that are broadly expressed across the background populations, isolating true, 
    lineage-specific signatures suitable for mixture deconvolution.

    Parameters
    ----------
    adata_pb
        An annotated data matrix containing pseudobulk count profiles.
    groupby
        The key of the observations grouping to consider for contrasts (e.g., cell types).
    layer
        The layer in `adata_pb` containing unnormalized, raw integer counts. PyDESeq2 
        strictly requires unnormalized data.
    subset_key
        Column defining broader categorical subsets. If provided, contrasts are computed 
        strictly within each subset. This prevents rare subpopulations from being 
        overwhelmed by distant lineages (e.g., finding CD4+ T cell markers only 
        against other T cells, rather than the entire immune system).
    design
        The explicit design formula (e.g., '~dataset + condition + cell_type'). 
        If `None`, defaults to `~groupby`.
    min_samples
        Minimum samples required to express a gene above `min_counts`. Biologically, 
        this should generally be set to the size of your smallest experimental group. 
        Setting this too low may cause the model to fit to outlier noise.
    min_counts
        Minimum normalized count threshold. Defaults to 5, which is deliberately lower 
        than traditional bulk RNA-seq thresholds to rescue biologically relevant 
        regulatory genes in rare, low-capture single-cell populations.
    rank_filter_method
        Method to preemptively drop highly sparse genes. Pass `"cholesky"` to enable 
        algebraic full-rank filtering. Highly recommended for sparse pseudobulk data 
        to prevent the generalized linear model from crashing on rank-deficient genes.
    gene_symbols
        Column in `adata_pb.var` to use for human-readable gene names.
    key_added
        The key in `adata_pb.uns` where results are saved if `inplace=True`.
    n_jobs
        Number of CPUs to use for parallelizing Wald tests.
    inplace
        If `True`, saves results directly to `adata_pb.uns[key_added]`. 
        If `False`, bypasses the AnnData object and returns the results dictionary.

    Returns
    -------
    If `inplace=True`, returns `None` and updates `adata_pb.uns[key_added]` with 
    the inverted results dictionary. 
    
    If `inplace=False`, returns a dictionary where keys are the DESeq2 metric 
    names (e.g., `'log2FoldChange'`, `'pvalue'`, `'names'`) and values are 
    pandas DataFrames containing the statistics for all groups.
    """
    try:
        import pydeseq2  # noqa: F401
    except ImportError as e:
        raise ImportError("`rank_genes_sum2zero` requires `pydeseq2`.") from e

    design_str = design or f"~{groupby}"
    if groupby not in design_str:
        raise ValueError(f"Design formula {design_str!r} must contain {groupby!r}.")

    # Track group sizes to detect if the user modified the input metadata,
    # this safely invalidates the cache if the cell annotations or arena mappings have changed.
    group_counts = adata_pb.obs[groupby].value_counts().to_dict()
    subset_counts = adata_pb.obs[subset_key].value_counts().to_dict() if subset_key else None

    if key_added in adata_pb.uns:
        cached = adata_pb.uns[key_added]
        c_params = cached.get("params", {})
        if (
            c_params.get("groupby") == groupby
            and c_params.get("design") == design_str
            and c_params.get("subset_key") == subset_key
            and c_params.get("rank_filter_method") == rank_filter_method
            and c_params.get("min_samples") == min_samples
            and c_params.get("min_counts") == min_counts
            and c_params.get("layer") == layer
            # Catch silent metadata mutations
            and c_params.get("group_counts") == group_counts
            and c_params.get("subset_counts") == subset_counts
        ):
            logg.info(
                f"found cached deseq2 results in `.uns[{key_added!r}]`. skipping computation"
            )
            return None if inplace else cached

    covariates = set(re.findall(r"\w+", design_str))
    covariates.add(groupby)
    if subset_key:
        covariates.add(subset_key)

    missing_cols = [c for c in covariates if c not in adata_pb.obs]
    if missing_cols:
        raise KeyError(f"Required columns {missing_cols} not found in `adata_pb.obs`.")

    is_na_df = pd.DataFrame({col: _is_na_strict(adata_pb.obs[col]) for col in covariates})
    if is_na_df.any().any():
        nan_cols = is_na_df.columns[is_na_df.any()].tolist()
        raise ValueError(
            f"Found missing values (NaN, None, or empty strings) in metadata columns: {nan_cols}. "
            f"PyDESeq2 strictly requires complete metadata to accurately calculate degrees of freedom. "
            f"You can identify the offending cells using `adata_pb.obs[{nan_cols}].isna()` and filter them."
        )

    logg.info(
        f"ranking genes using sum-to-zero deseq2 contrasts with design {design_str!r}"
    )
    start = time.time()

    wald_tests_per_group = {}
    deferred_warnings: dict[str, list[str]] = {}
    apply_cholesky_filter = rank_filter_method == "cholesky"

    if subset_key is None:
        logg.info("computing global one-vs-rest contrasts")
        results_df, group_warnings = _fit_deseq2_model(
            adata_pb,
            groupby,
            design_str,
            min_samples=min_samples,
            min_counts=min_counts,
            apply_cholesky_filter=apply_cholesky_filter,
            n_jobs=n_jobs,
            layer=layer,
        )
        wald_tests_per_group.update(results_df)
        if group_warnings:
            deferred_warnings["global"] = group_warnings

        elapsed = timedelta(seconds=int(time.time() - start))
        logg.info(f"finished deseq2 contrasts ({elapsed})")

    else:
        unique_subsets = adata_pb.obs[subset_key].dropna().unique()
        logg.info(
            f"computing contrasts within {len(unique_subsets)} subsets defined by {subset_key!r}"
        )

        skipped_singular = []
        skipped_single_group = []

        for ref_name in tqdm(unique_subsets, desc="testing subsets"):
            logg.debug(f"testing within subset: {ref_name}")

            # Explicit copy is required to safely mutate categorical levels downstream.
            # memory overhead is negligible because pseudobulk n_obs is usually small.
            adata_sub = adata_pb[adata_pb.obs[subset_key] == ref_name].copy()

            if adata_sub.obs[groupby].nunique() < 2:
                skipped_single_group.append(ref_name)
                continue

            for col in covariates:
                if isinstance(adata_sub.obs[col].dtype, pd.CategoricalDtype):
                    adata_sub.obs[col] = adata_sub.obs[
                        col
                    ].cat.remove_unused_categories()

            try:
                results_df, group_warnings = _fit_deseq2_model(
                    adata_sub,
                    groupby,
                    design_str,
                    min_samples=min_samples,
                    min_counts=min_counts,
                    apply_cholesky_filter=apply_cholesky_filter,
                    n_jobs=n_jobs,
                    layer=layer,
                )
                wald_tests_per_group.update(results_df)
                if group_warnings:
                    deferred_warnings[str(ref_name)] = group_warnings

            except np.linalg.LinAlgError:
                skipped_singular.append(ref_name)
                continue

        if skipped_single_group:
            logg.warning(
                f"skipped {len(skipped_single_group)} subsets because they contain fewer than 2 "
                f"unique categories in {groupby!r} ({', '.join(str(s) for s in skipped_single_group)})"
            )

        if skipped_singular:
            logg.warning(
                f"skipped {len(skipped_singular)} subsets due to singular design matrices. "
                "Try running with `rank_filter_method='cholesky'` to automatically drop the offending genes."
            )

    for subset_name, warnings_list in deferred_warnings.items():
        for w in set(warnings_list):
            logg.warning(f"pydeseq2 warning in subset {subset_name!r}: {w.strip()}")

    if not wald_tests_per_group:
        raise RuntimeError(
            "No contrasts could be computed. All subsets failed or were skipped."
        )

    gene_symbol_mapping = None
    if gene_symbols is not None:
        if gene_symbols in adata_pb.var:
            gene_symbol_mapping = adata_pb.var[gene_symbols].astype(str)
        else:
            logg.warning(
                f"did not find {gene_symbols!r} in `adata_pb.var`. falling back to `var_names`"
            )


    tidy_wald_tests = []

    for group, group_wald_df in wald_tests_per_group.items():
        assign_dict = {groupby: group}
        if gene_symbol_mapping is not None and gene_symbols is not None:
            has_symbol = group_wald_df["var_names"].isin(gene_symbol_mapping.index)
            assign_dict[gene_symbols] = np.where(
                has_symbol,
                group_wald_df["var_names"].map(gene_symbol_mapping),
                group_wald_df["var_names"],
            )

        group_wald_formatted = (
            group_wald_df.assign(**assign_dict)
            .sort_values("stat", ascending=False)
            .reset_index(drop=True)
        )
        tidy_wald_tests.append(group_wald_formatted)

    wald_tests_long = pd.concat(tidy_wald_tests, ignore_index=True)
    
    sanitize_dict = {"var_names": wald_tests_long["var_names"].fillna("").astype(str)}
    if gene_symbols is not None:
        sanitize_dict[gene_symbols] = wald_tests_long[gene_symbols].fillna("").astype(str)

    # Coerce metadata to pure string arrays to prevent h5ad serialization crashes
    # caused by jagged float NaNs during concatenation.
    wald_tests_long = (
        wald_tests_long.assign(**sanitize_dict)
        .astype({groupby: "category"})
    )
    cols = ["var_names", groupby]
    if gene_symbols is not None:
        cols.append(gene_symbols)
    cols.extend(["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"])
    
    wald_tests_long = wald_tests_long.loc[:, [c for c in cols if c in wald_tests_long.columns]].copy()

    deseq2_results_dict = {
        "results": wald_tests_long,
        "params": {
            "groupby": groupby,
            "group_counts": adata_pb.obs[groupby].value_counts().to_dict(),
            "design": design_str,
            "reference": "subset_key" if subset_key else "rest",
            "method": "pydeseq2_sum2zero",
            "subset_key": subset_key,
            "subset_counts": adata_pb.obs[subset_key].value_counts().to_dict() if subset_key else None,
            "rank_filter_method": rank_filter_method,
            "min_samples": min_samples,
            "min_counts": min_counts,
            "gene_symbols": gene_symbols,
            "layer": layer,
        },
    }

    if inplace:
        adata_pb.uns[key_added] = deseq2_results_dict
        logg.info(f"added to `.uns[{key_added!r}]`")
        return None

    return deseq2_results_dict
