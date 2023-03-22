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
