# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.9.13 ('shears_dev')
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.preprocessing
import pyreadr
from threadpoolctl import threadpool_limits, threadpool_info
import os

# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                            "intra_op_parallelism_threads=8")
import jax.numpy as jnp
from tqdm.contrib.concurrent import process_map
import itertools

# %%
threadpool_info()

# %%

threadpool_limits(8)

# %%
# adata = sc.read_h5ad("/home/sturm/projects/2020/pircher-scrnaseq-lung/data/20_build_atlas/add_additional_datasets/03_update_annotation/artifacts/full_atlas_merged.h5ad")
# adata = adata[adata.obs["origin"] == "tumor_primary", :]
# sc.pp.subsample(adata, n_obs=20000)
# adata.write_h5ad("data/luca_20k.h5ad")

# %%
tcga = pyreadr.read_r(
    "/home/sturm/projects/2020/pircher-scrnaseq-lung/data/13_tcga/for_scissor/nsclc_primary_tumor.rds"
)[None]
adata_tcga = sc.AnnData(tcga.T)

# %%
tcga_meta = pd.read_csv(
    "/home/sturm/projects/2020/pircher-scrnaseq-lung/tables/tcga/clinical_data_for_scissor.tsv",
    sep="\t",
).set_index("TCGA_patient_barcode")
adata_tcga.obs = adata_tcga.obs.join(tcga_meta)

# %%
adata_sc = sc.read_h5ad("data/luca_20k.h5ad")

# %%
common_genes = list(sorted(set(adata_tcga.var_names) & set(adata_sc.var_names)))

# %%
adata_tcga = adata_tcga[:, common_genes].copy()
adata_sc = adata_sc[:, common_genes].copy()

# %%
adata_sc.layers["quantile_norm"] = sklearn.preprocessing.quantile_transform(adata_sc.X)
adata_tcga.layers["quantile_norm"] = sklearn.preprocessing.quantile_transform(
    adata_tcga.X
)

# %%
adata_sc.layers["quantile_norm"]

# %%
# corr_result = np.corrcoef(adata_sc.layers["quantile_norm"].toarray(), adata_tcga.layers["quantile_norm"])

# %%
corr_result = jnp.corrcoef(
    adata_tcga.layers["quantile_norm"], adata_sc.layers["quantile_norm"].toarray()
)

# %%
corr_mat = corr_result[: adata_tcga.shape[0], adata_tcga.shape[0] :]

# %%
corr_mat.shape

# %%
import statsmodels.formula.api as smf
import statsmodels.api as sm

# %%
corr_iter = list(corr_mat[:, i] for i in range(corr_mat.shape[1]))


# %%
def check_cell(corr, df):
    df["corr"] = corr
    res = smf.glm(
        "C(type) ~ corr + C(gender) + age + C(tumor_stage_ajcc)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    return res.pvalues["corr"], res.params["corr"]


# %%
df = adata_tcga.obs.loc[
    :, ["type", "tp53_mutation", "gender", "age", "tumor_stage_ajcc"]
]

# %%
all_res = process_map(
    check_cell, corr_iter, itertools.repeat(df), chunksize=100, max_workers=32
)

# %%
all_res = np.array(all_res)

# %%
adata_sc.obs["pvalue_sign"] = -np.log10(all_res[:, 0]) * np.sign(all_res[:, 1])
adata_sc.obs["coef"] = all_res[:, 1]
adata_sc.obs["coef_signif"] = [
    x if p < 0.0001 else np.nan for x, p in zip(all_res[:, 1], all_res[:, 0])
]

# %%
sc.pl.umap(
    adata_sc,
    color=["pvalue_sign", "coef", "coef_signif"],
    cmap="bwr",
    vmin=-10,
    vmax=10,
)


# %%
sc.pl.umap(adata_sc, color=["cell_type"])

# %%
