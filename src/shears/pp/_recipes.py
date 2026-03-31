import warnings

from anndata import AnnData
from ._normalization import quantile_norm


def recipe_shears(adata_sc: AnnData, adata_bulk: AnnData, *, n_top_genes=2000, layer_sc=None, layer_bulk=None, key_added="quantile_norm"):
    """
    Automatically preprocess data for shears.

    Single-cell input data is expected to contain raw counts.
    Bulk data is expected to contain TPM (or any other measure metric for gene length).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", anndata.ImplicitModificationWarning)
        adata_sc = adata_sc[:, adata_sc.var_names.isin(adata_bulk.var_names)]
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_top_genes, layer=layer_sc, flavor="seurat_v3", subset=True)
        adata_bulk = adata_bulk[:, adata_sc.var_names]

    quantile_norm(adata_sc, layer=layer_sc, key_added=key_added)
    quantile_norm(adata_bulk, layer=layer_bulk, key_added=key_added)

    return adata_sc, adata_bulk
