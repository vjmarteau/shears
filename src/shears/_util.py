import logging

import pandas as pd
import statsmodels.stats.multitest
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map as tqdm_process_map


def _choose_mtx_rep(adata, use_raw=False, layer=None):
    """Get gene expression from anndata depending on use_raw and layer"""
    is_layer = layer is not None
    if use_raw and is_layer:
        raise ValueError(
            "Cannot use expression from both layer and raw. You provided:" f"'use_raw={use_raw}' and 'layer={layer}'"
        )
    if is_layer:
        return adata.layers[layer]
    elif use_raw:
        return adata.raw.X
    else:
        return adata.X


def process_map(fn, *iterables, **tqdm_kwargs):
    """Wrapper around tqdm.contrib.concurrent.process_map that doesn't use multiprocessing if max_workers = 1."""
    if tqdm_kwargs.get("max_workers", None) == 1:
        return list(tqdm_kwargs.get("tqdm_class", tqdm)(map(fn, *iterables), total=tqdm_kwargs.get("total", None)))
    else:
        return tqdm_process_map(fn, *iterables, **tqdm_kwargs)


def _parallelize_with_joblib(delayed_objects, *, total=None, **kwargs):
    """Wrapper around joblib.Parallel that shows a progressbar if the backend supports it.

    Progressbar solution from https://stackoverflow.com/a/76726101/2340703
    """
    try:
        return tqdm(Parallel(return_as="generator", **kwargs)(delayed_objects), total=total)
    except ValueError:
        logging.info(
            "Backend doesn't support return_as='generator'. No progress bar will be shown. "
            "Consider setting verbosity in joblib.parallel_config"
        )
        return Parallel(return_as="list", **kwargs)(delayed_objects)


def _cell_worker_map(cell_weights, worker, *, n_jobs = None, backend = "loky"):
    cell_names = list(cell_weights.index)
    weights_arr = cell_weights.to_numpy()
    jobs = (delayed(worker)(weights_arr[i, :]) for i in range(len(cell_names)))

    res_list = list(
        _parallelize_with_joblib(
            jobs,
            total=len(cell_names),
            n_jobs=n_jobs,
            backend=backend,
        )
    )
    return pd.DataFrame(res_list, index=cell_names, columns=["pvalue", "coef"])


def fdr_correction(df, pvalue_col="pvalue", *, key_added="fdr", inplace=False):
    """Adjust p-values in a data frame with test results using FDR correction."""
    if not inplace:
        df = df.copy()

    df[key_added] = statsmodels.stats.multitest.fdrcorrection(df[pvalue_col].values)[1]

    if not inplace:
        return df
