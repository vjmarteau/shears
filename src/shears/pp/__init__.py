from anndata import AnnData


def quantile_norm(adata: AnnData, layer=None, key_added="quantile_norm"):
    """Perform quantile normalization on AnnData object

    Stores the normalized data in a new layer with the key `key_added`.
    """
    raise NotImplementedError


def preprocess_shears(adata_sc, adata_bulk):
    """Automatically apply all preprocessing steps (highly variable genes, subsetting, quantile normalization).

    Convenient wrapper...
    """
    raise NotImplementedError


def cell_weights(adata_sc, adata_bulk, *, inplace=True, key_added="cell_weights"):
    """
    Computes a bulk_sample x cell matrix assigning each cell a weight for each bulk sample.

    This is conceptually similar to deconvolution, except that instead of a signature matrix,
    we have a single-cell dataset.

    If inplace is True, stores the resulting matrix in adata_bulk.obsm[key_added]
    """
    raise NotImplementedError
