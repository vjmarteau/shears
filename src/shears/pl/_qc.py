import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes

logg = logging.getLogger(__name__)


def signature_coexpression(
    adata_bulk: AnnData,
    features: Sequence[str],
    *,
    title: str | None = None,
    layer: str | None = None,
    gene_symbols: str | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    **kwargs,
) -> Axes | None:
    """\
    Visualize the co-expression correlation matrix of signature features.

    This function calculates Pearson correlation between features in the bulk 
    expression data. It serves as a visual companion to `prune_signatures`, 
    allowing users to visually validate that the selected signature features 
    form a biologically coherent, co-regulated module prior to deconvolution.

    Parameters
    ----------
    adata_bulk
        AnnData object containing bulk expression data.
    features
        List of feature identifiers (e.g., Ensembl IDs) to correlate. 
        Must correspond to `adata_bulk.var_names`.
    title
        Title of the plot. Defaults to "Signature co-expression".
    layer
        Layer in `adata_bulk` to use for calculations. If `None`, use `.X`.
    gene_symbols
        Column name in `adata_bulk.var` containing human-readable gene symbols. 
        If provided, these symbols will be used for plot labels.
    ax
        Pre-existing axes for the plot. If `None`, a new figure is created.
    show
        Whether to display the plot. If `False`, returns the axis object.
    **kwargs
        Additional arguments passed to `seaborn.heatmap`.

    Returns
    -------
    If `show=False`, returns the matplotlib axes object. Otherwise, returns `None`.
    """
    valid_features = adata_bulk.var_names.intersection(features).tolist()

    if len(valid_features) < 2:
        raise ValueError(
            "Insufficient valid features found in `adata_bulk` to compute correlation. "
            "At least 2 valid features are required."
        )

    adata_sub = adata_bulk[:, valid_features]
    X_sub = adata_sub.layers[layer] if layer else adata_sub.X

    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()

    # Align the math exactly with `prune_signatures`.
    # We suppress division-by-zero warnings for completely unexpressed "dropout" genes.
    # Seaborn will elegantly render these NaN correlations as blank white squares,
    # visually alerting the user that the gene is dead in the bulk tissue.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(X_sub, rowvar=False)

    plot_labels = valid_features
    if gene_symbols is not None:
        if gene_symbols not in adata_bulk.var.columns:
            raise KeyError(f"Column {gene_symbols!r} not found in `adata_bulk.var`.")

        symbol_map = adata_sub.var[gene_symbols].astype(str).to_dict()

        # Replace only if the symbol is valid and not an empty string or 'nan'
        plot_labels = [
            (
                symbol_map.get(f, f)
                if symbol_map.get(f, f).lower() not in ["", "nan"]
                else f
            )
            for f in valid_features
        ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    heatmap_params = dict(
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        xticklabels=plot_labels,
        yticklabels=plot_labels,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Pearson Correlation"},
    )
    heatmap_params.update(kwargs)

    sns.heatmap(corr_matrix, ax=ax, **heatmap_params)

    # Aesthetic polish & Explicit Labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    ax.set_xlabel("Signature Features", fontsize=10, fontweight="bold", labelpad=10)
    ax.set_ylabel("Signature Features", fontsize=10, fontweight="bold", labelpad=10)

    if title:
        ax.set_title(title, pad=15)
    elif not ax.get_title():
        ax.set_title("Signature Co-expression (Bulk Tissue)", pad=15)

    if show is None:
        show = True

    if show:
        plt.show()
        return None

    return ax
