import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

logg = logging.getLogger(__name__)


def _parallelize_with_joblib(
    delayed_objects: Sequence[Any], *, total: int | None = None, **kwargs: Any
) -> Any:
    """\
    Wrapper around joblib.Parallel that shows a progressbar if the backend supports it.

    Progressbar solution from https://stackoverflow.com/a/76726101/2340703
    """
    try:
        return tqdm(
            Parallel(return_as="generator", **kwargs)(delayed_objects), total=total
        )
    except ValueError:
        logg.info(
            "Backend doesn't support return_as='generator'. No progress bar will be shown. "
            "Consider setting verbosity in joblib.parallel_config"
        )
        return Parallel(return_as="list", **kwargs)(delayed_objects)


def _chunk_worker(
    worker_func: Callable[..., Any],
    weights_arr: np.ndarray,
    start: int,
    end: int,
    *args: Any,
    **kwargs: Any,
) -> tuple[int, int, np.ndarray]:
    """Process a batch of cells natively using pre-allocated chunk arrays."""
    # slice the array locally to leverage joblib's automatic memmapping.
    # see https://joblib.readthedocs.io/en/latest/parallel.html#working-with-numerical-data-in-shared-memory-memmapping
    weights_chunk = weights_arr[start:end, :]

    # Pre allocate (chunk.shape, pval, coef, se) to prevent dynamic resizing overhead in the worker process
    chunk_res = np.empty((weights_chunk.shape[0], 3), dtype=np.float64)

    for i, row in enumerate(weights_chunk):
        # Enforce minimum 3 sample representation to preserve positive degrees of freedom,
        # protecting lapack libraries from rank-deficient matrix segfaults.
        if np.count_nonzero(row) < 3:  
            chunk_res[i, :] = (1.0, 0.0, float("inf"))
        else:
            chunk_res[i, :] = worker_func(row, *args, **kwargs)

    return start, end, chunk_res


def _cell_worker_map(
    cell_weights: pd.DataFrame,
    worker_func: Callable[..., Any],
    *args: Any,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
    backend: str = "loky",
    **kwargs: Any,
) -> pd.DataFrame:
    """Map a worker function across single-cell arrays using memory-mapped multiprocessing."""
    cell_names = cell_weights.index

    # Force c-contiguous array to ensure zero-copy memmapping. This prevents joblib
    # from pickling/copying massive dataframes into the RAM of every worker.
    weights_arr = np.ascontiguousarray(cell_weights.to_numpy(), dtype=np.float64)
    n_cells = weights_arr.shape[0]

    if chunk_size is None:
        chunk_size = int(np.ceil(min(max(n_cells / 1000, 50), 1000)))

    intervals = [
        (i, min(i + chunk_size, n_cells)) for i in range(0, n_cells, chunk_size)
    ]

    jobs = (
        delayed(_chunk_worker)(worker_func, weights_arr, start, end, *args, **kwargs)
        for start, end in intervals
    )

    # Pre allocate main results array
    results_arr = np.empty((n_cells, 3), dtype=np.float64)

    # Shutdown the reusable executor to prevent c-level memory fragmentation 
    # and state contamination between independent modeling runs.
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)

    # TODO: Not sure max_tasks_per_child works as I want it to ...
    # max_tasks_per_child guarantees workers are preemptively killed and recycled.
    # this clears underlying openblas/lapack memory fragmentation and prevents
    # transient hardware segfaults on mathematically noisy single-cell chunks.
    for start, end, chunk_res in _parallelize_with_joblib(
        jobs,
        total=len(intervals),
        n_jobs=n_jobs,
        backend=backend,
        backend_kwargs={"max_tasks_per_child": 1},
    ):
        results_arr[start:end, :] = chunk_res

    return pd.DataFrame(results_arr, index=cell_names, columns=["pvalue", "coef", "se"])
