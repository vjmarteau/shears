def shears(adata_sc, adata_bulk, *, inplace=True, key_added="shears"):
    """
    Compute score for each cell.

    Stores the scores in adata_sc.obs[key_added] unless inplace is False.
    """
    raise NotImplementedError
