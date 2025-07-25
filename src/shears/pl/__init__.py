from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_shears_boxplots(
    res: pd.DataFrame,
    *,
    negative_label: str = "",
    positive_label: str = "",
    exclude_ns: bool = False,
    save_plot: bool = False,
    prefix: str = "",
    save_dir: Optional[str] = None,
    file_ext: str = "pdf",
    display: bool = True,
) -> plt.Figure:

    df = res.copy()

    if exclude_ns:
        df = df[df["level"] != "n.s."]
        df["group"] = df["group"].astype(str)
    
    score_df = df.assign(
        mean=df.groupby("group", observed=False)["mean_weight"].transform("mean"),
        std=df.groupby("group", observed=False)["mean_weight"].transform("std"),
        lower_bound=lambda x: x["mean"] - 3 * x["std"],
        upper_bound=lambda x: x["mean"] + 3 * x["std"],
        is_outlier=lambda x: (x["mean_weight"] < x["lower_bound"])
        | (x["mean_weight"] > x["upper_bound"]),
    )

    mean_df_col = score_df.groupby("group", observed=True)["mean_weight"].median()
    norm = mcolors.TwoSlopeNorm(vmin=mean_df_col.min(), vmax=mean_df_col.max(), vcenter=0)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    group_ns = score_df.loc[score_df["level"] == "n.s.", "group"].unique().tolist()
    group_col_dict = {
        g: "darkgrey" if g in group_ns else cmap(norm(mean_df_col.loc[g]))
        for g in mean_df_col.index
    }

    order = (
        score_df.loc[~score_df["is_outlier"]]
        .groupby("group", observed=True)["mean_weight"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(x=0, c="grey")
    sns.boxplot(
        y="group",
        x="mean_weight",
        data=score_df.loc[~score_df["is_outlier"]],
        order=order,
        ax=ax,
        hue="group",
        palette=group_col_dict,
        linewidth=1,
        fliersize=1,
        width=0.8,
        dodge=False,
        orient="h",
    )

    ax.text(-1e6, -0.8, negative_label, ha="right")
    ax.text(1e6, -0.8, positive_label, ha="left")

    if save_plot:
        plt.savefig(
            f"{save_dir}/{prefix}-shears_boxplot.{file_ext}",
            bbox_inches="tight",
        )
        if display:
            plt.show()
        plt.close()
