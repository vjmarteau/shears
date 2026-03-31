from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData


def rank_genes_sum2zero_df(
    adata_pb: AnnData,
    subset_groups: str | Sequence[str] | None = None,
    *,
    key: str = "rank_genes_sum2zero",
    signatures_key: str | None = None,
) -> pd.DataFrame:
    """\
    Extract DESeq2 sum-to-zero results in a long-format DataFrame.
    """
    if key not in adata_pb.uns:
        raise KeyError(
            f"Could not find {key!r} in `adata_pb.uns`. Please run the corresponding tool first."
        )

    results_dict = adata_pb.uns[key]
    df = results_dict["results"].copy()
    params = results_dict.get("params", {})
    groupby_col = params.get("groupby", "group")

    if subset_groups is not None:
        target_groups = [subset_groups] if isinstance(subset_groups, str) else list(subset_groups)
        available_groups = df[groupby_col].unique().tolist()
        
        if missing := set(target_groups) - set(available_groups):
            raise ValueError(f"Requested groups {missing} not found in `.uns[{key!r}]`.")
            
        df = df[df[groupby_col].isin(target_groups)].copy()

    if signatures_key is not None:
        if signatures_key not in adata_pb.uns:
            raise KeyError(
                f"Could not find signatures {signatures_key!r} in `adata_pb.uns`."
            )
        sig_dict = adata_pb.uns[signatures_key].get("signatures", {})
        
        approved_genes = set(gene for genes in sig_dict.values() for gene in genes)
        df = df[df["var_names"].isin(approved_genes)].copy()

    # Prevent -inf during downstream -log10(p-value) transformations for plotting
    for p_col in ["pvalue", "padj"]:
        if p_col in df.columns:
            df[p_col] = df[p_col].replace({0.0: np.nextafter(0, 1)})

    df.attrs.update(params)
    if signatures_key:
        df.attrs["filtered_by"] = signatures_key

    return df.reset_index(drop=True)


def obs_df(
    adata: AnnData,
    keys: str | Sequence[str],
    obsm_key: str | None = None,
) -> pd.DataFrame:
    """\
    Retrieve cell-level data from AnnData's obs or obsm attributes.

    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        A single key or a sequence of keys to retrieve from `adata.obs` 
        or `adata.obsm[obsm_key]`.
    obsm_key
        If provided, also look for the keys in this specific DataFrame 
        stored within `adata.obsm`.

    Returns
    -------
    A Pandas DataFrame containing the requested cell-level covariates, 
    indexed by `adata.obs_names`.
    """
    keys_list = [keys] if isinstance(keys, str) else list(keys)
    df = pd.DataFrame(index=adata.obs_names)

    for key in keys_list:
        if key in adata.obs.columns:
            df[key] = adata.obs[key]

        elif obsm_key is not None and obsm_key in adata.obsm:
            obsm_data = adata.obsm[obsm_key]
            if isinstance(obsm_data, pd.DataFrame) and key in obsm_data.columns:
                df[key] = obsm_data[key]
            else:
                raise KeyError(f"Key {key!r} not found in `adata.obsm[{obsm_key!r}]`.")
        else:
            raise KeyError(
                f"Key {key!r} could not be found in `adata.obs` or `obsm_key` {obsm_key!r}."
            )

    return df


def signatures(
    adata_pb: AnnData,
    group: str | None = None,
    *,
    key: str = "shears_signatures",
) -> list[str] | dict[str, list[str]]:
    """\
    Safely retrieve extracted or merged signatures from the AnnData object.

    Parameters
    ----------
    adata_pb
        The annotated data matrix.
    group
        The specific cell type / lineage to retrieve. If `None`, returns a 
        dictionary of all signatures.
    key
        The `.uns` key where the signatures are stored.

    Returns
    -------
    A list of gene symbols if `group` is provided. Otherwise, a dictionary 
    mapping all groups to their respective gene lists.
    """
    if key not in adata_pb.uns:
        raise KeyError(
            f"Could not find {key!r} in `adata_pb.uns`. "
            "Please run `sh.tl.extract_signatures` or `sh.tl.merge_signatures` first."
        )
    
    sig_dict = adata_pb.uns[key].get("signatures", {})

    if group is None:
        return sig_dict

    if group not in sig_dict:
        raise ValueError(
            f"Group {group!r} not found in the signatures. "
            f"Available groups: {list(sig_dict.keys())}"
        )

    return sig_dict[group]
