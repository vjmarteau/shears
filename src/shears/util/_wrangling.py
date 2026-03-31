import logging

import pandas as pd
from anndata import AnnData

logg = logging.getLogger(__name__)


def _is_na_strict(series: pd.Series) -> pd.Series:
    """\
    Identify missing values, optimizing for categorical arrays to prevent memory spikes.
    """
    is_na_mask = series.isna()

    if isinstance(series.dtype, pd.CategoricalDtype):
        # O(k) operation on the unique categories instead of O(N) on the full array
        invalid_categories = [
            c
            for c in series.cat.categories
            if str(c).strip().lower() in {"", "nan", "none", "unknown", "na"}
        ]
        if invalid_categories:
            is_na_mask |= series.isin(invalid_categories)
    elif series.dtype == object:
        series_str = series.astype(str).str.strip().str.lower()
        is_na_mask |= series_str.isin({"", "nan", "none", "unknown", "na"})

    return is_na_mask


def _get_clean_mask(adata_bulk: AnnData, obs_columns: list[str]) -> pd.Series:
    """\
    Dynamically scan for metadata artifacts and return a boolean mask.
    Returns a mask instead of a sliced AnnData View to prevent ImplicitModificationWarnings
    when tools write intermediate outputs.
    """
    is_na_mask = pd.Series(False, index=adata_bulk.obs_names)
    for col in obs_columns:
        is_na_mask |= _is_na_strict(adata_bulk.obs[col])

    n_missing = is_na_mask.sum()
    if n_missing > 0:
        logg.warning(
            f"found {n_missing} cells with missing or 'unknown' values in columns {obs_columns}; "
            "dynamically excluding them from the mathematical calculation."
        )
    return ~is_na_mask
