import itertools

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm

from shears._util import process_map


def _test_cell(formula, df):
    mod = smf.glm(formula, data=df, family=sm.families.Binomial())
    res = mod.fit()
    return res.pvalues["cell_weight"], res.params["cell_weight"]


def shears(
    adata_sc,
    adata_bulk,
    *,
    dep_var,
    covariate_str="",
    inplace=True,
    cell_weights_key="cell_weights",
    key_added="shears",
    n_jobs=None,
):
    """
    Compute score for each cell.

    Stores the scores in adata_sc.obs[key_added] unless inplace is False.

    Parameters
    ----------
    covariate_str
        covariates to add, e.g. `+ age + C(gender)`
    """
    formula = f"{dep_var} ~ cell_weight {covariate_str}"
    keep_cols = [c for c in adata_bulk.obs.columns if c in dep_var + covariate_str]
    assert "cell_weight" not in keep_cols, "cell_weight is a reserved column name"
    bulk_obs = adata_bulk.obs.loc[:, keep_cols].copy()

    print(formula)

    def _df_iter():
        for c in adata_sc.obs_names:
            tmp_df = bulk_obs.copy()
            tmp_df["cell_weight"] = adata_sc.obsm[cell_weights_key].loc[c, adata_bulk.obs_names]
            yield tmp_df

    res = process_map(
        _test_cell,
        itertools.repeat(formula),
        _df_iter(),
        total=adata_sc.shape[0],
        chunksize=100,
        max_workers=n_jobs,
        tqdm_class=tqdm,
    )

    return pd.DataFrame(res, index=adata_sc.obs_names, columns=["pvalue", "coef"])
