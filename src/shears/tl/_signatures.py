import logging
import warnings
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from shears.get import rank_genes_sum2zero_df
from shears.util._wrangling import _get_clean_mask, _is_na_strict

from ._utils import (
    _calculate_jaccard_overlap,
    _extract_bipartite_signatures,
    _extract_standard_signatures,
    _get_connected_components,
    _get_sparse_or_dense_agg,
)

logg = logging.getLogger(__name__)


def check_collinearity(
    adata_pb: AnnData,
    *,
    groupby: str = "cell_type",
    top_n: int = 500,
    overlap_threshold: float = 0.70,
    layer: str | None = None,
    key_added: str = "collinearity",
    inplace: bool = True,
) -> dict[str, Any] | None:
    """\
    Evaluate baseline transcriptional overlap to flag collinearity risks for deconvolution.

    In shears, highly overlapping cell types passed as independent fine-grained 
    signatures may suffer from L2 penalty smearing during Ridge Regression. This tool 
    calculates internal depth-normalized expression profiles (CPM) to identify which 
    populations share massive baseline similarities.

    Biological Rationale for Defaults:
        * top_n=500: Captures the primary identity footprint and highly expressed 
          lineage markers while avoiding the noisy, lowly-expressed dropout zone.
        * overlap_threshold=0.70: Populations sharing >70% of their top abundant 
          transcriptome are typically the same fundamental state (or severely 
          cross-contaminated) and are highly prone to collinearity.

    Parameters
    ----------
    adata_pb
        An annotated data matrix containing pseudobulk count profiles.
    groupby
        The metadata column defining the fine-grained populations.
    top_n
        The number of highly abundant transcripts to evaluate for baseline overlap.
    overlap_threshold
        The Jaccard threshold (0.0 to 1.0) at which populations are flagged for risk.
    layer
        The expression layer to use (must contain raw counts). If `None`, uses `.X`.
    key_added
        The column name added to `.uns` containing the suggested cluster assignments. 
    inplace
        If `True`, saves results to `adata_pb.uns[key_added]`. If `False`, returns the dict.

    Returns
    -------
    If `inplace=True`, returns `None`. Otherwise, returns a dictionary containing the 
    overlap report and merged components.
    """
    logg.info(
        f"evaluating baseline overlap on top {top_n} transcripts for collinearity risk"
    )

    clean_mask = _get_clean_mask(adata_pb, [groupby])
    counts_matrix = adata_pb.layers[layer] if layer else adata_pb.X
    valid_obs = adata_pb.obs[clean_mask]

    group_sums = {}
    for group in valid_obs[groupby].unique():
        group_mask = (valid_obs[groupby] == group).values

        group_sums[group] = _get_sparse_or_dense_agg(
            counts_matrix[clean_mask], group_mask, "sum"
        )

    df_sums = pd.DataFrame(group_sums, index=adata_pb.var_names).T

    # Strictly depth-normalize to CPM to prevent library size bias from skewing abundance
    df_cpm = df_sums.div(df_sums.sum(axis=1), axis=0) * 1e6

    report_df, network_edges = _calculate_jaccard_overlap(
        df_cpm, top_n, overlap_threshold
    )
    raw_components = _get_connected_components(
        nodes=list(df_cpm.index), edges=network_edges
    )

    merged_components = [comp for comp in raw_components if len(comp) > 1]
    safe_components = [", ".join(comp) for comp in merged_components]

    if safe_components:
        logg.warning(
            f"detected {len(safe_components)} highly overlapping group(s) "
            f"(>= {overlap_threshold*100:.0f}% overlap). consider assigning a "
            f"shared coarse anchor or merging these in the "
            f"single-cell reference prior to pseudobulking: {'; '.join(safe_components)}"
        )
    else:
        logg.info("no highly overlapping groups detected. all types appear distinct")

    res_dict = {
        "overlap_report": report_df,
        "components": safe_components,
        "params": {
            "groupby": groupby,
            "top_n": top_n,
            "overlap_threshold": overlap_threshold,
            "layer": layer,
        },
    }

    if inplace:
        adata_pb.uns[key_added] = res_dict
        logg.info(f"added collinearity report to `.uns[{key_added!r}]`")
        return None

    return res_dict


def flag_undetectable_markers(
    adata_bulk: AnnData,
    adata_pb: AnnData,
    *,
    min_bulk_mean: float | None = None,
    min_bulk_quantile: float | None = 0.1,
    layer_bulk: str | None = "ref_scaled",
    key_added: str = "bulk_detectable",
    inplace: bool = True,
) -> pd.Series | None:
    """\
    Flag genes whose signal is too weak to be reliably detected in bulk mixtures.

    In bulk deconvolution, marker genes must be "loud" enough to survive dilution. 
    If a gene is highly specific to a rare cell type but expressed at very low absolute 
    levels, its signal in the bulk mixture will be buried in ambient sequencing noise. 
    This function establishes a global noise floor based on the bulk data's distribution
    and creates a boolean mask of markers that rise above it.

    Parameters
    ----------
    adata_bulk
        An annotated data matrix containing the target bulk count profiles.
    adata_pb
        The annotated data matrix containing the single-cell reference. 
        The resulting mask is mapped to `adata_pb.var_names`.
    min_bulk_mean
        An absolute minimum average expression required to keep a gene.
    min_bulk_quantile
        A dynamic, scale-agnostic threshold (0.0 to 1.0). For example, `0.10` drops 
        any gene falling in the bottom 10% of the bulk tissue's global expression. 
        Takes precedence over `min_bulk_mean` if both are provided.
    layer_bulk
        The layer in `adata_bulk` containing depth-normalized counts (e.g., CPM). 
        Defaults to `"ref_scaled"`. If not found, safely falls back to `.X`.
    key_added
        The column name added to `adata_pb.var` if `inplace=True`.
    inplace
        If `True`, saves the boolean mask directly to `adata_pb.var[key_added]`. 
        If `False`, returns the boolean pandas Series.

    Returns
    -------
    If `inplace=True`, returns `None` and updates `adata_pb.var`. 
    If `inplace=False`, returns a boolean `pd.Series` mapping `adata_pb.var_names` 
    to their pass/fail status.
    """
    # We calculate the absolute mean expression across all bulk samples.
    # To be useful for downstream Ridge Regression, a single-cell marker
    # must demonstrate that its signal is loud enough to consistently rise
    # above the technical noise floor of the target bulk sequencing platform.
    if layer_bulk is not None:
        if layer_bulk in adata_bulk.layers:
            X_bulk = adata_bulk.layers[layer_bulk]
        else:
            logg.warning(
                f"layer {layer_bulk!r} not found in `adata_bulk.layers`. "
                "falling back to `.X`. Ensure your data is normalized!"
            )
            X_bulk = adata_bulk.X
    else:
        X_bulk = adata_bulk.X

    if sp.issparse(X_bulk):
        mean_bulk_expr = np.asarray(X_bulk.mean(axis=0)).flatten()
    else:
        mean_bulk_expr = np.asarray(X_bulk).mean(axis=0)

    # Dynamically calculate the noise floor based on the bulk
    # dataset's unique distribution, making the filter immune to linear vs. log
    # scaling differences.
    if min_bulk_quantile is not None and (0.0 <= min_bulk_quantile <= 1.0):
        dynamic_threshold = float(np.quantile(mean_bulk_expr, min_bulk_quantile))
        logg.info(
            f"calculated dynamic detectability floor at quantile {min_bulk_quantile}: "
            f"{dynamic_threshold:.4f}"
        )
    else:
        dynamic_threshold = float(min_bulk_mean) if min_bulk_mean is not None else 0.0
        logg.info(f"using absolute detectability threshold: {dynamic_threshold:.4f}")

    bulk_abundance = pd.Series(mean_bulk_expr, index=adata_bulk.var_names)
    aligned_bulk_expr = bulk_abundance.reindex(adata_pb.var_names, fill_value=0.0)

    is_expressed_in_bulk = aligned_bulk_expr > dynamic_threshold

    n_passed = is_expressed_in_bulk.sum()
    n_total = len(is_expressed_in_bulk)

    logg.info(
        f"flagged {n_total - n_passed} weak markers. "
        f"{n_passed} / {n_total} candidate genes passed the detectability threshold."
    )

    if inplace:
        adata_pb.var[key_added] = is_expressed_in_bulk.values
        logg.info(f"saved detectability mask to `adata_pb.var[{key_added!r}]`")
        return None

    return is_expressed_in_bulk


def extract_signatures(
    adata_pb: AnnData,
    source_key: str,
    *,
    n_top_features: int = 50,
    padj_threshold: float = 0.1,
    min_lfc: float = 0.5,
    min_basemean: float | None = None,
    basemean_quantile: float | None = 0.5,
    method: Literal["standard", "bipartite"] = "standard",
    max_pool: int = 100,
    exclusive: bool = False,
    stable_features_key: str | None = "stable_features",
    key_added: str = "shears_signatures",
    inplace: bool = True,
) -> dict[str, Any] | None:
    """\
    Extract biologically stable, lineage-specific signatures from DESeq2 contrasts.

    This function isolates optimal marker genes for deconvolution by heavily penalizing 
    transcriptional noise. It establishes a dynamic expression floor to ignore dropout 
    artifacts and sorts candidates by their statistically confident effect size 
    (lower-bound fold-change) rather than raw fold-change.

    Parameters
    ----------
    adata_pb
        An annotated data matrix containing pseudobulk count profiles.
    source_key
        The key in `adata_pb.uns` where the raw DESeq2 statistics are stored.
    n_top_features
        Maximum number of marker genes to extract per lineage.
    padj_threshold
        Maximum adjusted p-value for a gene to be considered a candidate.
    min_lfc
        Minimum log2 fold-change for a gene to be considered a candidate.
    min_basemean
        Absolute minimum BaseMean expression threshold.
    basemean_quantile
        Dynamic minimum BaseMean expression threshold. Tuning this balances the 
        signal-to-noise ratio:
            * `0.8` (Top 20%): Prioritizes robust, highly abundant structural transcripts.
            * `0.3` (Top 70%): Discovers lowly expressed but specific regulators (e.g., TFs).
    method
        Extraction methodology. `"standard"` extracts the top global markers. `"bipartite"` 
        forces highly specific, mutually exclusive state assignment.
    max_pool
        The maximum rank depth to search when `method="bipartite"`.
    exclusive
        Whether to enforce mutually exclusive markers in the standard method.
    stable_features_key
        The `.var` column containing the boolean mask of genes robust enough to 
        survive bulk dilution. If provided, safely filters the candidate pool.
    key_added
        The key in `adata_pb.uns` where results will be saved.
    inplace
        If `True`, saves results to `adata_pb.uns[key_added]`. 
        If `False`, returns the results dictionary directly.
    """
    if source_key not in adata_pb.uns:
        raise KeyError(f"Source key {source_key!r} not found in `adata_pb.uns`.")

    params = adata_pb.uns[source_key].get("params", {})
    groupby = params.get("groupby", "group")
    logg.info(
        f"extracting {method} signatures from {source_key!r} (groupby={groupby!r})"
    )

    wald_tests_df = rank_genes_sum2zero_df(adata_pb, key=source_key)

    if stable_features_key is not None:
        if stable_features_key not in adata_pb.var:
            raise KeyError(
                f"Could not find mask {stable_features_key!r} in `adata_pb.var`. "
                "Run `flag_undetectable_markers` first, or set `stable_features_key=None`."
            )
        
        # Restrict the candidate pool to genes that are "loud" enough in the
        # bulk mixture to be reliably deconvolved, avoiding noise-fitting.
        valid_genes = adata_pb.var_names[adata_pb.var[stable_features_key]]
        wald_tests_df = wald_tests_df.loc[lambda x: x["var_names"].isin(valid_genes)].copy()

    # Sort by the lower confidence bound of the LFC (95% CI) to prioritize genes
    # that are both strongly upregulated AND stable across the bulk cohort.
    wald_tests_df["lfc_lower_bound"] = wald_tests_df["log2FoldChange"] - (
        1.96 * wald_tests_df["lfcSE"]
    )

    if basemean_quantile is not None and (0.0 < basemean_quantile <= 1.0):
        # Calculate against the unique background transcriptome to prevent
        # lineages with massive marker lists from skewing the distribution.
        universe = wald_tests_df.drop_duplicates(subset=["var_names"])
        dynamic_basemean_floor = float(universe["baseMean"].quantile(basemean_quantile))
        logg.info(
            f"calculated dynamic baseMean floor at quantile {basemean_quantile}: {dynamic_basemean_floor:.2f}"
        )
    else:
        dynamic_basemean_floor = (
            float(min_basemean) if min_basemean is not None else 0.0
        )

    passing_candidates = wald_tests_df.loc[
        lambda x: (x["padj"] <= padj_threshold)
        & (x["log2FoldChange"] >= min_lfc)
        & (x["baseMean"] >= dynamic_basemean_floor)
    ]

    if method == "bipartite":
        # Highly similar sub-states often fight over the same pan-markers. Bipartite
        # matching acts as a global optimizer, forcing the algorithm to distribute
        # markers so every lineage gets the most mutually exclusive fingerprint possible.
        logg.debug(f"running maximum weight bipartite matching (max_pool={max_pool})")
        signatures = _extract_bipartite_signatures(
            passing_candidates, groupby, n_top_features, max_pool
        )
    elif method == "standard":
        logg.debug("running standard top-N marker extraction")
        signatures = _extract_standard_signatures(
            passing_candidates, groupby, n_top_features, exclusive
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'standard' or 'bipartite'."
        )

    res_dict = {
        "signatures": signatures,
        "params": {
            "source_key": source_key,
            "groupby": groupby,
            "n_top_features": n_top_features,
            "padj_threshold": padj_threshold,
            "min_lfc": min_lfc,
            "min_basemean": min_basemean,
            "basemean_quantile": basemean_quantile,
            "dynamic_basemean_floor_applied": dynamic_basemean_floor,
            "method": method,
            "stable_features_key": stable_features_key,
        },
    }

    if inplace:
        adata_pb.uns[key_added] = res_dict
        logg.info(f"added extracted signatures to `.uns[{key_added!r}]`")
        return None
    return res_dict


def merge_signatures(
    adata_pb: AnnData,
    coarse_key: str,
    fine_key: str,
    *,
    n_anchors: int = 5,
    n_splitters: int = 25,
    min_bottleneck_quantile: float = 0.10,
    min_child_fraction: float = 0.75,
    layer: str | None = None,
    key_added: str = "shears_signatures",
    inplace: bool = True,
) -> dict[str, Any] | None:
    """\
    Merge broad lineage anchors with highly specific state splitters.

    In deconvolution, regression models struggle to assign weights when closely 
    related cell states (e.g., CD4+ and CD8+ T cells) share too many generic markers. 
    This function constructs a hierarchical "barcode" for each sub-type by combining:
    
    1. Anchors: Broad, lineage-defining genes that lock the model into the correct 
       macroscopic compartment (e.g., T cells vs. B cells).
    2. Splitters: Highly specific, mutually exclusive genes that force the model 
       to distribute weights correctly among the fine-grained states.

    Parameters
    ----------
    adata_pb
        The annotated data matrix containing pseudobulk profiles.
    coarse_key
        The `.uns` key containing the broad lineage markers (anchors).
    fine_key
        The `.uns` key containing the fine-grained state markers (splitters).
    n_anchors
        The number of overarching lineage anchors to append to each sub-type.
    n_splitters
        The maximum number of fine-grained splitters to assign to each sub-type's 
        final signature. If the upstream splitter extraction pulled a larger number 
        of genes (e.g., 50), setting this to a lower number (e.g., 25) creates an 
        "Exclusion Moat": all 50 genes are protected from being stolen as anchors, 
        but only the top 25 are assigned to the Ridge solver to maintain L2 penalty 
        balance across all cell populations.
    min_bottleneck_quantile
        The expression quantile used to establish a biological noise floor. Genes 
        falling below this in a given sub-type are considered unexpressed background.
    min_child_fraction
        The fraction of sub-types within a lineage that must express an anchor for 
        it to be considered a valid, stable representation of the whole compartment.
    layer
        If provided, uses `adata_pb.layers[layer]` for expression thresholding. 
        Otherwise, defaults to `adata_pb.X`.
    key_added
        The key in `adata_pb.uns` where results are saved if `inplace=True`.
    inplace
        Whether to save the result directly to `adata_pb.uns` or return the dictionary.

    Returns
    -------
    If `inplace=True`, returns `None` and updates `adata_pb.uns[key_added]`.
    If `inplace=False`, returns a dictionary containing the merged signatures.
    """
    coarse_params = adata_pb.uns[coarse_key]["params"]
    fine_params = adata_pb.uns[fine_key]["params"]
    coarse_groupby, fine_groupby = coarse_params["groupby"], fine_params["groupby"]

    logg.info(
        f"merging signatures: {coarse_groupby!r} (anchors) + {fine_groupby!r} (splitters)"
    )

    clean_mask = _get_clean_mask(adata_pb, [fine_groupby, coarse_groupby])
    mapping_df = adata_pb.obs.loc[clean_mask, [fine_groupby, coarse_groupby]]

    # Vectorized hierarchy mapping
    majority_map = (
        mapping_df.value_counts()
        .groupby(fine_groupby, observed=True)
        .idxmax()
        .apply(lambda x: x[1])
    )
    parent_to_children = (
        majority_map.reset_index(name=coarse_groupby)
        .groupby(coarse_groupby, observed=True)[fine_groupby]
        .apply(list)
        .to_dict()
    )

    coarse_sigs = adata_pb.uns[coarse_key].get("signatures", {})
    fine_sigs = {
        k: list(v) for k, v in adata_pb.uns[fine_key].get("signatures", {}).items()
    }

    n_splitters_extracted = fine_params.get("n_top_features", 50)

    # Monotypic lineages (e.g., Cancer cells, Mast cells) do not have sub-states,
    # meaning DESeq2 cannot run internal contrasts to generate specific "Splitter"
    # genes for them. However, if their top defining genes are not globally
    # registered as protected, hierarchical cells (e.g., Macrophages) wearing
    # "local lineage blinders" might steal those genes to use as their own splitters.
    #
    # To prevent this "Local Blindspot" vulnerability, we promote the top N
    # global anchors of monotypic lineages to act as pseudo-splitters. This forces
    # them into the `forbidden_pool`, building a global firewall that prevents
    # hierarchical cells from co-opting terminal lineage markers and causing
    # severe Ridge regression weight smearing.
    for coarse_group, candidate_genes in coarse_sigs.items():
        children = parent_to_children.get(coarse_group, [])
        if len(children) == 1:
            child = children[0]
            if child not in fine_sigs:
                fine_sigs[child] = candidate_genes[:n_splitters_extracted]
    
    counts_matrix = adata_pb.layers[layer] if layer else adata_pb.X

    final_signatures = {}
    total_evicted = 0

    # State tracking for biological logging
    log_monotypic = []
    log_zero_anchors = []
    log_starved = []

    logg.info("applying dynamic scale-agnostic bottleneck filter to global anchors")

    # Process both fine group annotations and coarse parents groups
    for coarse_group, candidate_genes in coarse_sigs.items():
        children = parent_to_children.get(coarse_group, [])
        if not children:
            continue

        # Restrict off-target forbidden pool
        forbidden_pool = {
            g
            for other, sigs in fine_sigs.items()
            if other not in children
            for g in sigs
        }

        valid_candidates = [
            g
            for g in candidate_genes
            if g not in forbidden_pool and g in adata_pb.var_names
        ]

        if len(children) == 1:
            child = children[0]
            target_len = n_anchors + n_splitters 
            final_signatures[child] = valid_candidates[:target_len]
            log_monotypic.append(child)
            continue

        child_means = {}
        noise_floors = {}

        # Calculate baseline expression to establish a noise floor per child state,
        # preventing systemic background noise from being selected as a stable anchor.
        for child in children:
            child_mask = (adata_pb.obs[fine_groupby] == child).values
            if not child_mask.any():
                continue
            mean_vec = _get_sparse_or_dense_agg(counts_matrix, child_mask, "mean")
            child_means[child] = mean_vec
            noise_floors[child] = float(np.quantile(mean_vec, min_bottleneck_quantile))

        gene_indices = [adata_pb.var_names.get_loc(g) for g in valid_candidates]

        # Vectorized bottleneck filter
        expr_data = {
            c: child_means[c][gene_indices] for c in children if c in child_means
        }
        if not expr_data:
            continue

        expr_df = pd.DataFrame(expr_data, index=valid_candidates)
        is_expressed = expr_df > pd.Series(noise_floors)
        passing_genes = is_expressed.sum(axis=1) / len(children) >= min_child_fraction

        valid_expr_df = expr_df[passing_genes]
        total_evicted += len(expr_df) - len(valid_expr_df)

        valid_anchors = (
            valid_expr_df.apply(
                lambda row: float(row[is_expressed.loc[row.name]].quantile(0.15)),
                axis=1,
            )
            .sort_values(ascending=False)
            .index.tolist()
        )

        if n_anchors:
            valid_anchors = valid_anchors[:n_anchors]

        for child in children:
            child_specific = [g for g in valid_anchors if is_expressed.loc[g, child]]
            child_splitters = fine_sigs.get(child, [])[:n_splitters] 
            final_signatures[child] = child_specific + child_splitters

            # Track states that have lost biological definition for the model
            if len(child_specific) == 0:
                log_zero_anchors.append(child)
            if len(child_splitters) == 0:
                log_starved.append(child)

    logg.info(f"bottleneck complete: evicted {total_evicted} unstable anchors")

    if log_monotypic:
        logg.info(
            f"bypassed {len(log_monotypic)} monotypic lineages "
            f"(e.g., {', '.join(log_monotypic[:3])})"
        )

    if log_zero_anchors:
        logg.warning(
            f"Biological Drift: {len(log_zero_anchors)} sub-types retained 0 shared "
            f"lineage anchors. The model may struggle to assign these."
        )

    if log_starved:
        logg.warning(
            f"Marker Starvation: {len(log_starved)} sub-types lack sufficient specific "
            f"splitters. Deconvolution weights for these states may be highly unstable."
        )

    if final_signatures:
        sig_lengths = {cell: len(genes) for cell, genes in final_signatures.items()}
        min_len = min(sig_lengths.values())
        max_len = max(sig_lengths.values())
        
        if min_len > 0: 
            ratio = max_len / min_len

            if ratio >= 2.5:
                min_cells = [c for c, l in sig_lengths.items() if l == min_len]
                max_cells = [c for c, l in sig_lengths.items() if l == max_len]
                
                logg.warning(
                    f"Signature imbalance detected! Max/Min ratio is {ratio:.1f}x "
                    f"(Max: {max_len} in {max_cells[0]}, Min: {min_len} in {min_cells[0]}). "
                    f"A ratio >= 2.5x can cause severe L2 penalty bias in Ridge Regression, "
                    f"systematically under-predicting the smaller signatures. "
                    f"Consider lowering `n_splitters` to act as an equalizer."
                )

    res_dict = {
        "signatures": final_signatures,
        "params": {
            "coarse_key": coarse_key,
            "fine_key": fine_key,
            "coarse_groupby": coarse_groupby,
            "fine_groupby": fine_groupby,
            "n_anchors": n_anchors,
            "n_splitters": n_splitters,
        },
    }

    if inplace:
        adata_pb.uns[key_added] = res_dict
        logg.info(f"added merged signatures to `.uns[{key_added!r}]`")
        return None
    return res_dict


def prune_signatures(
    adata_bulk: AnnData,
    adata_pb: AnnData,
    *,
    signature_key: str = "shears_signatures",
    min_corr: float = 0.15,
    layer: str | None = None,
    key_added: str | None = None,
    inplace: bool = True,
) -> dict | None:
    """\
    Prune cell-type signatures based on median co-expression in target bulk data.

    Single-cell derived marker genes can suffer from the "Loud Neighbor" effect when 
    applied to bulk mixtures: a gene highly specific to a rare cell type in the reference 
    might actually be driven by a massive, off-target cell population in the bulk tissue. 
    By enforcing a minimum median Pearson correlation within the actual bulk mixture, 
    this function ensures the retained signature genes form a biologically coherent, 
    co-regulated module prior to deconvolution.

    Parameters
    ----------
    adata_bulk
        An annotated data matrix containing the target bulk count profiles.
    adata_pb
        The annotated data matrix containing the extracted single-cell signatures.
        This object serves strictly as a read-only reference.
    signature_key
        The key in `adata_pb.uns` where the gene signatures are stored.
    min_corr
        Minimum median Pearson correlation required to retain a gene.
    layer
        The layer in `adata_bulk` containing count profiles. If `None`, uses `X`.
    key_added
        The key in `adata_bulk.uns` where results are saved. If `None`, uses 
        `signature_key`.
    inplace
        If `True`, saves results directly to `adata_bulk.uns[key_added]`. 
        If `False`, bypasses the AnnData object and returns the results dictionary.

    Returns
    -------
    If `inplace=True`, returns `None` and updates `adata_bulk.uns`. 
    If `inplace=False`, returns a dictionary containing the pruned signatures.
    """
    if signature_key not in adata_pb.uns:
        raise KeyError(
            f"Could not find signatures in `adata_pb.uns[{signature_key!r}]`."
        )

    original_uns = adata_pb.uns[signature_key]
    signatures: dict[str, list[str]] = original_uns.get("signatures", {})

    if not signatures:
        raise ValueError(
            f"`adata_pb.uns[{signature_key!r}]` does not contain a valid 'signatures' dictionary."
        )

    filtered_signatures: dict[str, list[str]] = {}
    genes_pruned = 0
    signatures_modified = 0

    logg.info(f"evaluating bulk co-expression for {len(signatures)} signatures")

    # We validate signatures against the target bulk tissue to drop "loud neighbors"
    # genes specific in single-cell but ubiquitously expressed by a confounding,
    # dominating population in the bulk mixture.
    for target_group, candidate_genes in signatures.items():

        # Scverse idiom: native pd.Index intersection is safe, fast, and CoW-compliant
        valid_genes = adata_bulk.var_names.intersection(candidate_genes).tolist()

        if len(valid_genes) < 2:
            filtered_signatures[target_group] = valid_genes
            continue

        sub_adata = adata_bulk[:, valid_genes]
        X_sub = sub_adata.layers[layer] if layer else sub_adata.X

        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()

        # If a rare marker is completely absent in the bulk mixture,
        # its variance is zero, causing np.corrcoef to divide by zero. We catch and
        # ignore this expected mathematical edge-case to keep the logs clean.
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_matrix = np.corrcoef(X_sub, rowvar=False)
            np.fill_diagonal(corr_matrix, np.nan)

            # We rely on the median rather than the mean to ensure
            # the *majority* of the signature module moves synchronously, naturally
            # dropping isolated sub-modules that only correlate with one or two genes.
            median_corr = np.nanmedian(corr_matrix, axis=0)

        is_correlated = median_corr >= min_corr

        passed_genes = [
            g for g, is_corr in zip(valid_genes, is_correlated) if is_corr
        ]

        n_dropped = len(valid_genes) - len(passed_genes)
        if n_dropped > 0:
            genes_pruned += n_dropped
            signatures_modified += 1
            logg.debug(
                f"filtered out {n_dropped} uncoordinated genes from {target_group!r}"
            )

        filtered_signatures[target_group] = passed_genes

    save_key = key_added if key_added else signature_key

    if genes_pruned > 0:
        logg.info(
            f"dropped {genes_pruned} genes across {signatures_modified} groups "
            f"that failed the min_corr={min_corr} threshold"
        )
    else:
        logg.info("all signature genes passed co-expression validation")

    new_params = original_uns.get("params", {}).copy()
    new_params.update({"min_corr": min_corr, "layer_used": layer, "method": "median"})

    res_dict = {
        "signatures": filtered_signatures,
        "params": new_params,
    }

    if inplace:
        adata_bulk.uns[save_key] = res_dict
        logg.info(f"saved pruned signatures to `adata_bulk.uns[{save_key!r}]`")
        return None

    return res_dict


def score_signature_detectability(
    adata_bulk: AnnData,
    *,
    signature_key: str = "shears_signatures",
    layer: str | None = "ref_scaled",
    detectability_thresholds: tuple[float, float] = (0.25, 0.75),
) -> pd.DataFrame:
    """\
    Score the absolute bulk detectability of each cell-type signature.

    In deconvolution, cell types whose signature genes are barely detectable in 
    the bulk mixture will produce highly unstable, noise-driven regression weights. 
    This function calculates the median and mean bulk expression of each signature 
    module and translates the numerical detectability into an actionable biological 
    diagnostic (e.g., "Robust" vs. "High Risk").

    Parameters
    ----------
    adata_bulk
        An annotated data matrix containing the target bulk count profiles.
    signature_key
        The key in `adata_bulk.uns` where the gene signatures are stored.
    layer
        The layer in `adata_bulk` containing normalized expression data used
        for calculating detectability.
    detectability_thresholds
        A tuple of two floats `(lower_bound, upper_bound)`. 
        - Medians below `lower_bound` are flagged as "High Risk (Dilution)".
        - Medians above `upper_bound` are flagged as "Robust".
        - Values in between are flagged as "Moderate".

    Returns
    -------
    A pandas DataFrame containing detectability metrics and a categorical 
    `signal_quality` column, sorted from strongest to weakest signal.
    """
    if signature_key not in adata_bulk.uns:
        raise KeyError(
            f"Could not find signatures in `adata_bulk.uns[{signature_key!r}]`. "
            "Make sure you have run the signature extraction steps first."
        )

    signatures = adata_bulk.uns[signature_key].get("signatures", {})
    if not signatures:
        raise ValueError(f"`adata_bulk.uns[{signature_key!r}]` contains no signatures.")

    logg.info(f"scoring bulk signature detectability for {len(signatures)} groups")

    # Establish the global baseline expression of all genes in the
    # bulk tissue safely handling both dense and sparse representations.
    X_bulk = adata_bulk.layers[layer] if layer else adata_bulk.X
    if sp.issparse(X_bulk):
        mean_bulk_expr = np.asarray(X_bulk.mean(axis=0)).flatten()
    else:
        mean_bulk_expr = np.asarray(X_bulk).mean(axis=0)

    bulk_detectability = pd.Series(mean_bulk_expr, index=adata_bulk.var_names)

    records = []
    for target_group, candidate_genes in signatures.items():
        med_val, mean_val, n_genes = 0.0, 0.0, 0
        
        if candidate_genes:
            # Project the bulk expression onto the specific signature.
            # Genes missing from the bulk data are coerced to 0.0 to actively penalize 
            # the detectability score of markers that fail to capture in bulk sequencing.
            group_expr = bulk_detectability.reindex(candidate_genes, fill_value=0.0)
            med_val = float(group_expr.median())
            mean_val = float(group_expr.mean())
            n_genes = len(candidate_genes)
            
        records.append({
            "group": target_group,
            "median_detectability": med_val,
            "mean_detectability": mean_val,
            "n_genes": n_genes,
        })

    df_summary = (
        pd.DataFrame(records)
        .sort_values(by="median_detectability", ascending=False)
        .reset_index(drop=True)
    )

    lower_bound, upper_bound = detectability_thresholds
    
    df_summary["signal_quality"] = pd.cut(
        df_summary["median_detectability"],
        bins=[-np.inf, lower_bound, upper_bound, np.inf],
        labels=["High Risk (Dilution)", "Moderate", "Robust"],
        ordered=True,
    )

    df_summary["group"] = df_summary["group"].astype("category")

    return df_summary


def flag_bulk_detection_limits(
    adata_cohort: AnnData,
    *,
    min_abundance: float = 0.005,
    min_patients: int = 15,
    layer: str = "fractions",
    key_added: str = "bulk_detection_limits",
    copy: bool = False,
) -> AnnData | None:
    """\
    Flag cell populations that fall below the post-deconvolution bulk detection limit.

    Biological Rationale:
        This evaluates the true compositional fractions assigned by the Ridge Regression 
        to your finalized cell groups (defined by your `lineage_mapping` hierarchy). 
        It determines if a cell state actually survived the bulk tissue dilution. 
        
        It does NOT delete data. It adds a boolean mask to `.var` indicating which 
        input groups passed the detection limit. Users should use this mask to filter 
        their data before running downstream statistical inference (like Cox or GLMs) 
        to prevent zero-inflated artifacts.

    Parameters
    ----------
    adata_cohort
        The patient-level AnnData object containing post-deconvolution fractions.
    min_abundance
        The absolute bulk fraction (e.g., 0.005 for 0.5%) required to consider an input 
        cell group "detected" in a single patient's tumor.
    min_patients
        The minimum number of patients that must meet the `min_abundance` threshold.
    layer
        The layer containing unscaled, strictly compositional fractions (sums to 1).
    key_added
        The key added to `.uns` to store the evaluation report, and the prefix for 
        the `.var` boolean mask.

    Returns
    -------
    Updates `adata_cohort.var[f"passes_{key_added}"]` with a boolean mask.
    Updates `adata_cohort.uns[key_added]` with the full diagnostic report.
    """
    adata = adata_cohort.copy() if copy else adata_cohort

    params = {
        "min_abundance": min_abundance,
        "min_patients": min_patients,
        "layer": layer,
    }

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found. Save unscaled fractions first.")

    logg.info(
        f"evaluating bulk detection limits (>= {min_abundance*100}% in >= {min_patients} patients)"
    )

    X_fracs = adata.layers[layer]
    if sp.issparse(X_fracs):
        X_fracs = X_fracs.toarray()

    fractions_df = pd.DataFrame(X_fracs, index=adata.obs_names, columns=adata.var_names)

    patients_passing = (fractions_df >= min_abundance).sum(axis=0)
    detected_mask = patients_passing >= min_patients

    detected_pops = detected_mask[detected_mask].index.tolist()
    undetected_pops = detected_mask[~detected_mask].index.tolist()

    if undetected_pops:
        logg.warning(
            f"Flagged {len(undetected_pops)} hierarchical groups below bulk detection limit: "
            f"{', '.join(undetected_pops)}. Use `.var['passes_{key_added}']` to filter."
        )

    adata.uns[key_added] = {
        "detected_populations": detected_pops,
        "undetected_populations": undetected_pops,
        "params": params,
    }

    adata.var[f"passes_{key_added}"] = detected_mask.values
    logg.info(f"added boolean mask to `.var['passes_{key_added}']`")

    return adata if copy else None


def _blacklist(
    symbols: pd.Series,
    cell_cycle_genes: Iterable[str] | None = None,
    noise_regex: str | None = None,
    exclude_genes: Iterable[str] | None = None,
    retain_genes: Iterable[str] | None = None,
) -> pd.Series:
    """\
    Flag confounding or biologically noisy features.

    Identifies genes associated with the cell cycle, ubiquitous noise, or 
    cross-platform ambiguities (e.g., mitochondrial, ribosomal) that violate
    the linear additivity assumptions of the mixture model.
    """
    if cell_cycle_genes is None:
        # Tirosh et al., 2016 (Science). DOI: 10.1126/science.aad0501
        cell_cycle_genes = frozenset([
            "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", 
            "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", 
            "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", 
            "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", 
            "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1", 
            "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", 
            "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", 
            "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", 
            "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
            "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
            "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
            # Custom additions
            "ASPM", "CDKN3", "CCNA2", "STMN1", "MYO1E", "CDCA7L", "RFC4", "RANBP1", "BUB3", "CENPM",
            "KIF14", "KIF15", "PLK1", "MAD2L1",
        ])


    if noise_regex is None:
        # Ubiquitous metabolic hogs and library prep artifacts.
        # Mitigates RBC contamination (Hb) and capture biases (poly-A vs total RNA) for highly abundant transcripts.
        mito_ribo_hemo = r"^(?:MT-|RPS|RPL|MRPS|MRPL|HSP[A-Z0-9]+|DNAJ|HB[A-Z])"

        # Universal structural and housekeeping genes.
        # Prevents broad background expression from anchoring specific cell compartments.
        housekeeping_cyto = r"^(?:ACT[ABG]|TUB[AB]|GAPDH|B2M|EEF1A1|LMNA|VIM|TPM[0-9]+|MYH[0-9]+)"

        # scRNA-seq preparation (37°C enzymatic digestion, mechanical shear) 
        # induces massive transient stress responses not seen in snap-frozen bulk tumors. 
        # We ban these "state" genes so the model doesn't confuse processing trauma 
        # with stable cell-type identity:
        #
        # - NR4A Family: Upregulated by mechanical shear stress during dissociation.
        # - DDIT3/4 & GADD45: Driven by ischemia/hypoxia during surgical resection.
        # - UCP Family: Volatile metabolic/ROS sensors that fluctuate with tissue state.
        # - FOS/JUN/EGR/HSP: Standard Immediate Early Genes triggered by heat/digestion.
        stress_and_ieg = r"^(?:MALAT1|NEAT1|FOS[A-Z0-9]*|JUN[A-Z]?|EGR[0-9]+|NFKBI[A-Z]|TNFAIP[0-9]+|MT[0-9][A-Z]+|HERPUD1|SAT1|FTL|FTH1|DUSP[12]|PPP1R15A|ZFP36(?:L[12])?|ATF3|IER3|HIF1A|UCP[0-9]+|NR4A[1-3]|DDIT[34]|GADD45[ABG]|MXD1|ODC1|NAMPT|BID|ATG7|USP15|EML4|CYRIB)$"

        # Pan-tissue Interferon-Stimulated Genes (ISGs).
        # Banned to prevent the model from misinterpreting global tissue inflammation
        # (e.g., viral infection) as a shift in specific immune/stromal cell proportions.
        isg_response = r"^(?:IFI[A-Z0-9]*|ISG[0-9]+|OAS[A-Z0-9]*)$"

        # Patient-level covariates that drive massive transcriptional variance based on biological sex,
        # entirely independent of underlying cell type proportions.
        sex_markers = r"^(?:XIST|RPS4Y1|EIF1AY|DDX3Y|KDM5D|USP9Y)$"

        # Highly polymorphic genes whose expression is driven by global tissue inflammation (e.g., IFN-gamma exposure)
        # or patient-specific genetic haplotypes (HLA and KIR families) rather than stable, intrinsic cell lineage identity.
        hla_and_kir = r"^(?:HLA-|KIR[23][DS][A-Z0-9]+)"

        # Hypervariable immune clonotypes. Expression is idiosyncratic to the patient's specific repertoire
        # and won't map linearly. (Note: Regex safely allows constant regions [e.g., IGHM, TRAC] to pass as robust lineage anchors).
        vdj_receptors = r"^(?:TR[ABGD][VDJ]|IG[HKL][VDJ])"

        # Poorly annotated loci, BAC clones, and small RNAs (snoRNAs/miRNAs). Highly prone to multi-mapping errors
        # and massive poly-A vs Total RNA capture biases.
        ambiguous_loci = r"^(?:LOC[0-9]|LINC[0-9]|C[0-9]+orf[0-9]|SNORD[0-9]+|MIR[0-9]+|KCNQ1OT1|MIAT|ENSG[0-9]+)"
        bac_clones = r"^(?:A[CLPB][0-9]{5,}\.)"

        # Excluded due to dual biases: S-phase cell cycle dependency (marks cell division, not identity)
        # and massive capture rate discrepancies (canonical histones lack poly-A tails).
        histones = r"^(?:HIST[1-3]H|H2[AB][A-Z0-9]+|H3[A-Z0-9]+|H4[A-Z0-9]+|H[1-4]-|H[1-3]F)"

        noise_regex = f"{mito_ribo_hemo}|{housekeeping_cyto}|{stress_and_ieg}|{isg_response}|{sex_markers}|{hla_and_kir}|{vdj_receptors}|{ambiguous_loci}|{bac_clones}|{histones}"

    is_cycling = symbols.isin(set(cell_cycle_genes))
    is_noise = symbols.str.contains(noise_regex, regex=True, na=False)

    is_excluded = pd.Series(False, index=symbols.index)
    if exclude_genes is not None:
        is_excluded = symbols.isin(set(exclude_genes))

    is_blacklisted = is_cycling | is_noise | is_excluded

    if retain_genes is not None:
        is_retained = symbols.isin(set(retain_genes))
        is_blacklisted &= ~is_retained

    return is_blacklisted


def _calculate_dispersion(
    X_bulk: sp.spmatrix | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """\
    Calculate mean and dispersion from a bulk matrix.
    """
    if sp.issparse(X_bulk):
        mean_bulk = np.asarray(X_bulk.mean(axis=0)).ravel()
        mean_sq = np.asarray(X_bulk.power(2).mean(axis=0)).ravel()
        var_bulk = mean_sq - (mean_bulk**2)
    else:
        mean_bulk = np.asarray(X_bulk).mean(axis=0).ravel()
        var_bulk = np.asarray(X_bulk).var(axis=0).ravel()

    # Add epsilon (1e-6) to denominator to prevent ZeroDivisionError for perfectly sparse genes
    dispersion = var_bulk / (mean_bulk + 1e-6)

    return mean_bulk, dispersion


def flag_stable_features(
    adata_bulk: AnnData,
    adata_pb: AnnData,
    *,
    gene_symbols: str | None = None,
    min_bulk_quantile: float = 0.10,
    max_bulk_quantile: float = 0.995,
    max_dispersion_quantile: float = 0.99,
    cell_cycle_genes: Iterable[str] | None = None,
    noise_regex: str | None = None,
    exclude_genes: Iterable[str] | None = None,
    retain_genes: Iterable[str] | None = None,
    layer_bulk: str | None = "ref_scaled",
    key_added: str = "stable_features",
    inplace: bool = True,
) -> pd.Series | None:
    """\
    Flag stable features for non-negative ridge regression.

    Evaluates the bulk matrix to establish mathematical boundaries (expression floor, 
    ceiling, and dispersion outliers) and applies a biological blacklist 
    (e.g., cell cycle, ribosomal, hemoglobin genes) to the single-cell reference.

    Parameters
    ----------
    adata_bulk
        Target bulk mixtures to query statistical boundaries.
    adata_pb
        Single-cell reference pseudobulk. The final mask maps to its `var_names`.
    gene_symbols
        Column in `adata_pb.var` containing HGNC gene symbols. If `None`, 
        `var_names` is used.
    min_bulk_quantile
        Lower threshold for mean expression to drop ambient noise.
    max_bulk_quantile
        Upper threshold for mean expression to drop ubiquitous library hogs.
    max_dispersion_quantile
        Upper threshold for the index of dispersion to drop erratic spikes.
    cell_cycle_genes
        List of proliferation genes to ban. Defaults to Tirosh et al. (2016).
    noise_regex
        Regex to ban ubiquitous noise (e.g., mitochondria, ribosomes, stress lncRNAs).
    exclude_genes
        Exact gene symbols to forcibly drop regardless of statistical stability.
    retain_genes
        Exact gene symbols to forcibly save regardless of biological gating.
    layer_bulk
        Normalized layer to use in the bulk object.
    key_added
        Column name added to `adata_pb.var` containing the boolean mask.
    inplace
        Whether to add the mask to `adata_pb.var` or return it directly.

    Returns
    -------
    If `inplace=True`, returns `None` and adds a boolean column `{key_added}` 
    to `adata_pb.var` and execution parameters to `adata_pb.uns['{key_added}_params']`.
    Otherwise, returns the boolean mask as a :class:`~pandas.Series`.
    """
    logg.info(
        "evaluating global feature space for biological and statistical stability"
    )

    if adata_bulk.n_vars < 10000:
        logg.warning(
            f"`adata_bulk` contains only {adata_bulk.n_vars} features. "
            "Using a pre-filtered matrix skews statistical quantile boundaries "
            "and drops unaligned features. Consider passing the unfiltered transcriptome."
        )

    if gene_symbols is not None:
        if gene_symbols not in adata_pb.var:
            raise KeyError(f"column {gene_symbols!r} not found in `adata_pb.var`.")
        target_symbols = adata_pb.var[gene_symbols].copy()
    else:
        target_symbols = pd.Series(adata_pb.var_names, index=adata_pb.var_names)

    is_na_symbols = _is_na_strict(target_symbols)
    n_missing_symbols = is_na_symbols.sum()

    if n_missing_symbols == len(target_symbols):
        raise ValueError(
            "all gene symbols are missing or invalid. check the `gene_symbols` column."
        )
    elif n_missing_symbols > 0:
        logg.warning(
            f"found {n_missing_symbols} invalid/missing gene symbols; "
            "dynamically excluding them from the stable features mask"
        )

    target_symbols = target_symbols.astype(str)

    is_blacklisted = _blacklist(
        target_symbols, cell_cycle_genes, noise_regex, exclude_genes, retain_genes
    )
    is_blacklisted |= is_na_symbols
    is_biologically_safe = ~is_blacklisted

    logg.info(f"blacklisted {is_blacklisted.sum()} confounding features")

    X_bulk = adata_bulk.layers[layer_bulk] if layer_bulk else adata_bulk.X
    mean_bulk, dispersion = _calculate_dispersion(X_bulk)

    floor_limit = float(np.quantile(mean_bulk, min_bulk_quantile))
    ceiling_limit = float(np.quantile(mean_bulk, max_bulk_quantile))
    disp_limit = float(np.quantile(dispersion, max_dispersion_quantile))

    is_detectable = mean_bulk > floor_limit
    is_not_hog = mean_bulk <= ceiling_limit
    is_not_erratic = dispersion <= disp_limit

    is_statistically_safe = is_detectable & is_not_hog & is_not_erratic

    # Reindex is required because X_bulk features and adata_pb features may not strictly perfectly overlap
    bulk_safe_series = pd.Series(is_statistically_safe, index=adata_bulk.var_names)
    aligned_stat_safe = bulk_safe_series.reindex(adata_pb.var_names, fill_value=False)

    logg.info(
        f"flagged {(~is_detectable).sum()} noise genes (<{min_bulk_quantile}), "
        f"{(~is_not_hog).sum()} library hogs (>{max_bulk_quantile}), and "
        f"{(~is_not_erratic).sum()} erratic outliers (>{max_dispersion_quantile})"
    )

    final_mask = is_biologically_safe & aligned_stat_safe
    n_passed = final_mask.sum()
    logg.info(f"retained {n_passed} out of {len(adata_pb.var_names)} features")

    params = {
        "gene_symbols": gene_symbols,
        "min_bulk_quantile": min_bulk_quantile,
        "max_bulk_quantile": max_bulk_quantile,
        "max_dispersion_quantile": max_dispersion_quantile,
        "exclude_genes": list(exclude_genes) if exclude_genes else None,
        "retain_genes": list(retain_genes) if retain_genes else None,
        "layer_bulk": layer_bulk,
        "n_passed": int(n_passed),
    }

    if not inplace:
        return final_mask

    adata_pb.var[key_added] = final_mask.values
    adata_pb.uns[f"{key_added}_params"] = params
    logg.info(f"saved mask to `adata_pb.var[{key_added!r}]`")

    return None


def score_signature_cohesion(
    adata_bulk: AnnData,
    adata_pb: AnnData,
    *,
    signature_key: str = "shears_signatures",
    layer: str | None = None,
    key_added: str = "signature_cohesion",
    copy: bool = False,
) -> AnnData | None:
    """\
    Calculate the unidimensionality (cohesion) of gene signatures using PCA.

    Parameters
    ----------
    adata_bulk
        An annotated data matrix containing the target bulk count profiles.
    adata_pb
        The annotated data matrix containing the extracted single-cell signatures.
    signature_key
        The key in `adata_pb.uns` where the gene signatures are stored.
    layer
        The layer in `adata_bulk` containing count profiles. If `None`, uses `X`.
    key_added
        The key in `adata_pb.uns` where the resulting PCA metrics DataFrame is saved.
    copy
        If `True`, returns a copied `AnnData` object instead of modifying in place.

    Returns
    -------
    Returns the modified `AnnData` object if `copy=True`, otherwise returns `None`.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("`score_signature_cohesion` requires `scikit-learn`.") from e

    if copy:
        adata_pb = adata_pb.copy()

    if signature_key not in adata_pb.uns:
        raise KeyError(
            f"Could not find signatures in `adata_pb.uns[{signature_key!r}]`."
        )

    signatures = adata_pb.uns[signature_key].get("signatures", {})

    x_mat = adata_bulk.layers[layer] if layer else adata_bulk.X
    is_sparse = hasattr(x_mat, "toarray")

    records = []

    logg.info("calculating pca-based cohesion metrics for gene signatures")

    for group, genes in signatures.items():
        valid_features = [g for g in genes if g in adata_bulk.var_names]
        if len(valid_features) < 3:
            continue

        gene_indices = [adata_bulk.var_names.get_loc(g) for g in valid_features]
        x_sub = (
            x_mat[:, gene_indices].toarray() if is_sparse else x_mat[:, gene_indices]
        )

        pca = PCA(n_components=2)
        pca.fit(x_sub)

        pc1_var = pca.explained_variance_ratio_[0]
        pc2_var = pca.explained_variance_ratio_[1]

        records.append(
            {
                "group": group,
                "n_genes": len(valid_features),
                "pc1_variance": round(pc1_var, 3),
                "pc2_variance": round(pc2_var, 3),
                "cohesion_ratio": (
                    round(pc1_var / pc2_var, 2) if pc2_var > 0 else np.inf
                ),
            }
        )

    df = (
        pd.DataFrame(records)
        .sort_values("cohesion_ratio", ascending=True)
        .reset_index(drop=True)
    )

    adata_pb.uns[key_added] = {
        "cohesion": df,
        "params": {
            "signature_key": signature_key,
            "layer_used": layer,
        },
    }

    logg.info(f"added cohesion metrics to `.uns[{key_added!r}]`")

    return adata_pb if copy else None
