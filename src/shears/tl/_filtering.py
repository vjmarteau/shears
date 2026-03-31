import logging

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

logg = logging.getLogger(__name__)


def filter_unmapped_cells(
    adata_sc: AnnData,
    *,
    obsm_key: str = "cell_weights",
    min_samples: int = 3,
    inplace: bool = True,
) -> np.ndarray | None:
    """\
    Flag single cells that fail to map to the bulk target cohort.

    Cells are considered "unmapped" if they receive a deconvolution weight of 
    exactly zero in the vast majority of bulk target samples. In the context of 
    non-negative Ridge regression, this indicates the solver rejected the cell's 
    expression profile as biologically uninformative for reconstructing the bulk 
    tumor microenvironment. 

    By default, cells are retained if they receive a non-zero weight in at least 
    3 bulk samples, matching standard scverse heuristics for distinguishing 
    reproducible biological states from technical noise or donor-specific artifacts.

    Parameters
    ----------
    adata_sc
        The annotated data matrix of single cells.
    obsm_key
        The key in `adata_sc.obsm` containing the calculated cell weight matrix.
    min_samples
        The absolute minimum number of bulk target samples in which a cell must 
        receive a weight greater than 0 to be considered valid. Defaults to 3.
    inplace
        If `True`, adds a boolean mask to `adata_sc.obs['nonzero_bulk_weight']`. 
        If `False`, bypasses object mutation and returns the boolean mask directly.

    Returns
    -------
    If `inplace=True`, returns `None` and adds a boolean mask to 
    `adata_sc.obs['nonzero_bulk_weight']`.
        
    If `inplace=False`, returns a boolean `np.ndarray` of cells to keep.
    """
    if obsm_key not in adata_sc.obsm:
        raise KeyError(
            f"Key {obsm_key!r} not found in `adata_sc.obsm`. "
            "Please run `sh.pp.cell_weights` first."
        )

    weights_arr = adata_sc.obsm[obsm_key]
    if sp.issparse(weights_arr):
        keep_mask = np.asarray((weights_arr > 0).sum(axis=1)).flatten() >= min_samples
    else:
        keep_mask = (weights_arr > 0).sum(axis=1) >= min_samples

    n_dropped = adata_sc.n_obs - keep_mask.sum()

    if n_dropped > 0:
        logg.info(
            f"flagged {n_dropped} unmapped cells (weights < {min_samples} bulk samples) "
            "in `.obs['nonzero_bulk_weight']`"
        )

    if inplace:
        adata_sc.obs["nonzero_bulk_weight"] = keep_mask
        return None

    return keep_mask
