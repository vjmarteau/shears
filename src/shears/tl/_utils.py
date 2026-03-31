import itertools

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def _get_connected_components(
    nodes: list[str], edges: list[tuple[str, str]]
) -> list[list[str]]:
    """\
    Extract connected components from an edge list using Breadth-First Search.
    """
    adj = {n: [] for n in nodes}
    for n1, n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n1)

    visited = set()
    components = []
    for n in nodes:
        if n not in visited:
            comp = []
            q = [n]
            visited.add(n)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            components.append(comp)
    return components


def _calculate_jaccard_overlap(
    df_cpm: pd.DataFrame, top_n: int, overlap_threshold: float
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """\
    Calculate pairwise Jaccard similarity on the most highly expressed transcripts.
    """
    unique_groups = list(df_cpm.index)

    # Fast top-N extraction using numpy vectorization
    top_features_per_group = {
        group: set(
            df_cpm.columns[
                (
                    np.argpartition(row, -top_n)[-top_n:]
                    if len(row) > top_n
                    else np.arange(len(row))
                )
            ]
        )
        for group, row in zip(unique_groups, df_cpm.values, strict=False)
    }

    overlap_records = []
    network_edges = []

    for c1, c2 in itertools.combinations(unique_groups, 2):
        s1, s2 = top_features_per_group[c1], top_features_per_group[c2]
        overlap = len(s1 & s2) / len(s1 | s2) if (s1 or s2) else 0.0

        overlap_records.append(
            {
                "Group_A": c1,
                "Group_B": c2,
                "Shared_Abundant_Genes": len(s1 & s2),
                "Overlap_Fraction": float(overlap),
                "Collinearity_Risk": (
                    "High (Consider Shared Anchor or Merge)"
                    if overlap >= overlap_threshold
                    else "Low (Likely Distinct)"
                ),
            }
        )

        if overlap >= overlap_threshold:
            network_edges.append((c1, c2))

    report_df = (
        pd.DataFrame(overlap_records)
        .sort_values("Overlap_Fraction", ascending=False)
        .reset_index(drop=True)
    )
    report_df["Group_A"] = report_df["Group_A"].astype("category")
    report_df["Group_B"] = report_df["Group_B"].astype("category")
    report_df["Collinearity_Risk"] = report_df["Collinearity_Risk"].astype("category")

    return report_df, network_edges


def _extract_standard_signatures(
    passing_candidates: pd.DataFrame, groupby: str, n_top_features: int, exclusive: bool
) -> dict[str, list[str]]:
    """\
    Extract the top highly confident markers per group, optionally enforcing strict exclusivity.
    """
    top_markers = (
        passing_candidates.sort_values("lfc_lower_bound", ascending=False)
        .groupby(groupby, sort=False, observed=True)
        .head(n_top_features)
    )

    if exclusive:
        # Strictly drop any marker that appears in more than one lineage's top-N list
        top_markers = top_markers[
            ~top_markers.duplicated(subset=["var_names"], keep=False)
        ]

    return {
        group: df["var_names"].tolist()
        for group, df in top_markers.groupby(groupby, sort=False, observed=True)
    }


def _extract_bipartite_signatures(
    candidates: pd.DataFrame,
    groupby: str,
    n_top_features: int,
    max_pool: int,
) -> dict[str, list[str]]:
    """\
    Extract mutually exclusive marker features using maximum weight bipartite matching.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("scipy is required for bipartite matching extraction.") from e

    candidates = (
        candidates.sort_values("lfc_lower_bound", ascending=False)
        .groupby(groupby, sort=False, observed=True)
        .head(max_pool)
    )

    groups = candidates[groupby].unique()
    unique_genes = candidates["var_names"].unique()

    num_groups = len(groups)
    num_slots = num_groups * n_top_features
    num_genes = len(unique_genes)

    if num_genes == 0:
        return {str(group): [] for group in groups}

    gene_to_idx = {gene: i for i, gene in enumerate(unique_genes)}
    penalty = 1e9
    cost_matrix = np.full((num_slots, num_genes), penalty, dtype=float)

    for i, group in enumerate(groups):
        group_data = candidates[candidates[groupby] == group]
        gene_indices = group_data["var_names"].map(gene_to_idx).values
        weights = group_data["lfc_lower_bound"].values

        row_start = i * n_top_features
        row_end = (i + 1) * n_top_features
        cost_matrix[row_start:row_end, gene_indices] = -weights

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    signatures = {str(group): [] for group in groups}

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost < 0:
            group_idx = r // n_top_features
            group = groups[group_idx]
            gene = unique_genes[c]
            signatures[str(group)].append(gene)

    return signatures


def _get_sparse_or_dense_agg(x_mat, mask: np.ndarray, agg_func: str) -> np.ndarray:
    """\
    Calculate sum or mean across an axis natively without converting full matrix to dense array.
    """
    x_sub = x_mat[mask, :]
    if issparse(x_sub):
        res = getattr(x_sub, agg_func)(axis=0)
        return np.asarray(res).flatten()
    return np.asarray(getattr(x_sub, agg_func)(axis=0)).flatten()


def _broadcast(val: Any, length: int, name: str) -> list[Any]:
    """\
    Broadcast scalar parameters to match the number of keys.
    """
    if isinstance(val, str) or not isinstance(val, Sequence):
        return [val] * length
    if len(val) != length:
        raise ValueError(
            f"Length mismatch for {name!r}. Expected {length}, got {len(val)}."
        )
    return list(val)
