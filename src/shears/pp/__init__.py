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


def _block_iter(n_cells: int, n_bulk: int, block_size: int) -> Iterator[Tuple[int, int, int, int]]:
    """
    Compute the ranges of rows and columns for dividing an (n_cells × n_bulk) matrix into equally sized square chunks.
    Each block is defined by (i0, i1, j0, j1).
    """
    for i0 in range(0, n_cells, block_size):
        i1 = min(i0 + block_size, n_cells)
        for j0 in range(0, n_bulk, block_size):
            j1 = min(j0 + block_size, n_bulk)
            yield i0, i1, j0, j1


def _compute_block(sc_mat: np.ndarray, bulk_mat: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> List[Tuple[float, float, int, int]]:
    """
    Calculate the Pearson correlation coefficient and its p-value for each (cell, bulk) pair in the given block.
    Returns a list of tuples (r, p, global_cell_idx, global_bulk_idx).
    """
    res = []
    for ii in range(i0, i1):
        for jj in range(j0, j1):
            r, p = pearsonr(sc_mat[ii], bulk_mat[jj])
            res.append((r, p, ii, jj))
    return res


def cell_corrs(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    *,
    inplace: bool = True,
    layer_sc: str = "quantile_norm",
    layer_bulk: str = "quantile_norm",
    key_added: str = "pearson",
    n_jobs: int | None = None,
    backend: str = "loky",
    block_size: int | None = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Compute Pearson correlation & p-value between each single cell and each bulk sample
    using a 2D block-wise approach as in scirpy https://github.com/scverse/scirpy/blob/8bdd07b1448d99a4482d9b29eda0d0d93e659459/src/scirpy/ir_dist/metrics.py#L231-L233
    """

    sc_mat = adata_sc.layers[layer_sc]
    if sparse.issparse(sc_mat):
        sc_mat = sc_mat.toarray()

    bulk_mat = adata_bulk.layers[layer_bulk]
    if sparse.issparse(bulk_mat):
        bulk_mat = bulk_mat.toarray()

    n_cells = sc_mat.shape[0]
    n_bulk = bulk_mat.shape[0]

    # Dynamically adjust the block size such that there are ~1000 blocks within a range of 50 and 5000
    problem_size = n_cells * n_bulk
    if block_size is None:
        block_size = int(np.ceil(min(max(np.sqrt(problem_size / 1000), 50), 5000)))
    logging.info(f"cell_corrs block size set to {block_size}")

    # Precompute blocks as list to have total number of blocks for progressbar
    blocks = list(_block_iter(n_cells, n_bulk, block_size))

    tasks = (
        delayed(_compute_block)(sc_mat, bulk_mat, i0, i1, j0, j1)
        for (i0, i1, j0, j1) in blocks
    )

    block_results = list(
        _parallelize_with_joblib(
            tasks,
            total=len(blocks),
            n_jobs=n_jobs,
            backend=backend,
        )
    )

    corr_arr = np.zeros((n_cells, n_bulk), dtype=np.float32)
    pval_arr = np.zeros((n_cells, n_bulk), dtype=np.float32)

    for block in block_results:
        for r, p, ii, jj in block:
            corr_arr[ii, jj] = r
            pval_arr[ii, jj] = p

    corr_df = pd.DataFrame(
        corr_arr, index=adata_sc.obs_names, columns=adata_bulk.obs_names
    )
    pval_df = pd.DataFrame(
        pval_arr, index=adata_sc.obs_names, columns=adata_bulk.obs_names
    )

    if inplace:
        adata_sc.obsm[f"{key_added}_corr"] = corr_df
        adata_sc.obsm[f"{key_added}_pval"] = pval_df
    else:
        return corr_df, pval_df
