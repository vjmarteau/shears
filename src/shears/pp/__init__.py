import itertools
import logging
import warnings
from typing import Iterator, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.preprocessing
from anndata import AnnData
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from shears._util import fdr_correction, _parallelize_with_joblib


def quantile_norm(adata: AnnData, *, layer=None, key_added="quantile_norm", **kwargs):
    """Perform quantile normalization on AnnData object

    Stores the normalized data in a new layer with the key `key_added`.
    """
    X = adata.X if layer is None else adata.layers[layer]
    adata.layers[key_added] = sklearn.preprocessing.quantile_transform(X, **kwargs)


def recipe_shears(adata_sc, adata_bulk, *, n_top_genes=2000, layer_sc=None, layer_bulk=None, key_added="quantile_norm"):
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


def _deconvolute(bulk_sample, sc_mat, alpha, random_state):
    with threadpool_limits(limits=1):
        model = Ridge(alpha=alpha, positive=True, random_state=random_state)
        fit = model.fit(sc_mat, bulk_sample)
    return fit.coef_


def cell_weights(
    adata_sc,
    adata_bulk,
    *,
    inplace=True,
    alpha_callback=lambda adata_sc: adata_sc.shape[0],
    layer_sc="quantile_norm",
    layer_bulk="quantile_norm",
    key_added="cell_weights",
    random_state=0,
    n_jobs=None,
    backend="loky",
) -> Optional[pd.DataFrame]:
    """
    Computes a bulk_sample x cell matrix assigning each cell a weight for each bulk sample.

    This is conceptually similar to deconvolution, except that instead of a signature matrix,
    we have a single-cell dataset.

    If inplace is True, stores the resulting matrix in adata_sc.obsm[key_added]
    """

    assert all(adata_sc.var_names == adata_bulk.var_names), "var_names are not in the same order or are not the same"

    # convert sparse bulk layer to dense 1D arrays -> when obs > 1000 scanpy seems to auto convert to sparse ...
    if sparse.issparse(adata_bulk.layers[layer_bulk]):
        adata_bulk.layers[layer_bulk] = adata_bulk.layers[layer_bulk].toarray()

    jobs = (
        delayed(_deconvolute)(
            adata_bulk.layers[layer_bulk][i, :],
            adata_sc.layers[layer_sc].T,
            alpha_callback(adata_sc),
            random_state,
        )
        for i in range(adata_bulk.shape[0])
    )

    weights_list = list(
        _parallelize_with_joblib(
            jobs, total=adata_bulk.shape[0], n_jobs=n_jobs, backend=backend
        )
    )

    res = pd.DataFrame(
        np.array(weights_list).T, index=adata_sc.obs_names, columns=adata_bulk.obs_names
    )

    if inplace:
        adata_sc.obsm[key_added] = res
    else:
        return res_df
