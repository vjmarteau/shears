import itertools
import warnings
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.preprocessing
from anndata import AnnData
from sklearn.linear_model import Ridge
from sklearn.svm import NuSVR
from threadpoolctl import threadpool_limits

# TODO widget doesn't work in vscode :(
# from tqdm.auto import tqdm
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def quantile_norm(adata: AnnData, *, layer=None, key_added="quantile_norm"):
    """Perform quantile normalization on AnnData object

    Stores the normalized data in a new layer with the key `key_added`.
    """
    X = adata.X if layer is None else adata.layers[layer]
    adata.layers[key_added] = sklearn.preprocessing.quantile_transform(X)


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


@threadpool_limits.wrap(limits=1)
def _deconvolute_ridge(bulk_sample, sc_mat, alpha, random_state):
    model = Ridge(alpha=alpha, positive=True, random_state=random_state)
    fit = model.fit(sc_mat, bulk_sample)
    return fit.coef_


@threadpool_limits.wrap(limits=1)
def _deconvolute_nusvr(bulk_sample, sc_mat, alpha, random_state):
    model = NuSVR(C=10000**2, kernel="linear")
    fit = model.fit(sc_mat, bulk_sample)
    # for whatever reason, this returns a sparse matrix that happens to be dense (probably because the input is sparse)
    return fit.coef_.toarray()[0]


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
    method="ridge",
) -> Optional[pd.DataFrame]:
    """
    Computes a bulk_sample x cell matrix assigning each cell a weight for each bulk sample.

    This is conceptually similar to deconvolution, except that instead of a signature matrix,
    we have a single-cell dataset.

    If inplace is True, stores the resulting matrix in adata_sc.obsm[key_added]
    """
    f = {"ridge": _deconvolute_ridge, "nusvr": _deconvolute_nusvr}[method]
    res = process_map(
        f,
        (adata_bulk.layers[layer_bulk][i, :] for i in range(adata_bulk.shape[0])),
        itertools.repeat(adata_sc.layers[layer_sc].T),
        itertools.repeat(alpha_callback(adata_sc)),
        itertools.repeat(random_state),
        max_workers=n_jobs,
        chunksize=10,
        tqdm_class=tqdm,
        total=adata_bulk.shape[0],
    )

    res = pd.DataFrame(np.array(res).T, index=adata_sc.obs_names, columns=adata_bulk.obs_names)

    if inplace:
        adata_sc.obsm[key_added] = res
    else:
        return res
