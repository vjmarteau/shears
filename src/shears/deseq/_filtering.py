import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

logg = logging.getLogger(__name__)


def filter_genes_deseq2(
    adata_bulk: AnnData,
    *,
    min_samples: int = 3,
    min_counts: int = 10,
    layer: str | None = None,
    n_jobs: int | None = None,
    subset: bool = False,
    inplace: bool = True,
) -> tuple[np.ndarray, pd.Series] | None:
    """\

    Filter lowly expressed genes based on DESeq2 size factors (poscounts).
    
    Calculates size factors using the 'poscounts' estimator to robustly handle
    genes that have zero counts in some samples. This acts as an independent,
    non-specific pre-filter to improve computational speed and statistical power
    prior to differential expression analysis. Genes are retained only if they
    have at least `min_counts` normalized reads in at least `min_samples` samples.

    Inspired by `R DESeq2 <https://support.bioconductor.org/p/65256/>`_.

    Parameters
    ----------
    adata_bulk
        The annotated data matrix of shape `n_obs` × `n_vars`.
    min_samples
        Minimum number of samples required to express the gene above `min_counts`. 
        For highly stratified data, this is typically the size of the smallest 
        experimental group. For large, unstratified clinical bulk cohorts, a 
        robust heuristic is the square root of the total cohort size 
        (e.g., `int(np.sqrt(adata.n_obs))`).
    min_counts
        Minimum normalized count threshold.
    layer
        If provided, uses `adata_bulk.layers[layer]`. Otherwise, uses `adata_bulk.X`.
    n_jobs
        Number of CPUs to use for parallelizing DESeq2.
    subset
        Inplace subset to passing genes if `True`, otherwise merely indicate 
        passing genes via `.var['deseq2_keep']`.
    inplace
        Whether to place calculated metrics in `.obs` and `.var`, or return them.

    Returns
    -------
    If `inplace=True`, returns `None` and updates `adata_bulk`:
        - `adata_bulk.obs['deseq2_size_factors']`: Estimated size factors.
        - `adata_bulk.var['deseq2_keep']`: Boolean indicator of kept genes (if `subset=False`).
    
    If `inplace=False`, returns a tuple:
        - `keep_mask` (np.ndarray): Boolean mask of genes to keep.
        - `size_factors` (pd.Series): Estimated size factors.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
    except ImportError as e:
        raise ImportError(
            "Using `filter_genes_deseq2` requires `pydeseq2`. "
            "Install it with `pip install pydeseq2`."
        ) from e

    is_quiet = logg.getEffectiveLevel() > logging.DEBUG
    counts_matrix = adata_bulk.X if layer is None else adata_bulk.layers[layer]

    # Create lightweight temporary pydeseq2 object to avoid mutating the users main object or design matrices.
    adata_tmp = AnnData(counts_matrix)
    adata_tmp.obs_names = adata_bulk.obs_names

    # Size factor estimation does not care about the design matrix, putting intercept.
    # use poscounts to safely handle independent datasets with varying dropouts/zeros.
    dds = DeseqDataSet(adata=adata_tmp, design="~1", n_cpus=n_jobs, quiet=is_quiet)
    dds.fit_size_factors(fit_type="poscounts")

    if sp.issparse(dds.X):
        # Scale sparse rows by the inverse of size factors via dot product
        # to prevent dense broadcasting memory issues.
        inverse_size_factors = 1.0 / dds.obs["size_factors"].values
        scaled_counts = sp.diags(inverse_size_factors).dot(dds.X)

        # Modifying the sparse data array in-place avoids materializing a massive
        # dense boolean matrix during the threshold check.
        scaled_counts.data = np.where(scaled_counts.data >= min_counts, 1, 0)
        scaled_counts.eliminate_zeros()

        keep_mask = scaled_counts.getnnz(axis=0) >= min_samples
    else:
        # Fallback for dense arrays, scale the threshold instead of dividing the matrix.
        scaled_thresholds = min_counts * dds.obs["size_factors"].values[:, None]
        keep_mask = (dds.X >= scaled_thresholds).sum(axis=0) >= min_samples

    keep_mask = np.asarray(keep_mask).flatten()
    n_dropped = len(keep_mask) - keep_mask.sum()

    if n_dropped > 0:
        if subset:
            logg.info(
                f"filtered out {n_dropped} genes with fewer than {min_counts} "
                f"counts in {min_samples} samples"
            )
        else:
            logg.info(
                f"flagged {n_dropped} lowly expressed genes "
                f"(<{min_counts} counts in <{min_samples} samples) "
                "in `.var['deseq2_keep']`"
            )

    if inplace:
        adata_bulk.obs["deseq2_size_factors"] = dds.obs["size_factors"].copy()

        if subset:
            # TODO!
            # AnnData does not provide a public method for in-place subsetting.
            # Using the private `_inplace_subset_var` is the only way to drop genes
            # and free memory without making a full copy of the object, mirroring
            # the behavior of `scanpy.pp.filter_genes` and other scverse tools.
            # Not sure how I feel about using a private function from a dependency
            # that could be deprecated or changed without warning!
            adata_bulk._inplace_subset_var(keep_mask)
        else:
            adata_bulk.var["deseq2_keep"] = keep_mask

        return None

    return keep_mask, dds.obs["size_factors"].copy()
