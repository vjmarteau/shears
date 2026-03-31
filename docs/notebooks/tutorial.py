# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: shears-dev kernel
#     language: python
#     name: shears-dev
# ---

# %% [markdown]
# # Single-cell deconvolution and clinical association modeling with Shears
#
# `shears` is a computational framework that bridges the gap between single-cell resolution and bulk clinical cohorts. The algorithm operates in two distinct phases:
#
# 1. **Deconvolution:** It translates bulk RNA-seq tissue into relative cellular weights using optimized, lineage-specific gene signatures and non-negative Ridge regression.
# 2. **Clinical association modeling:** It links those predicted weights directly to patient phenotypes using highly parallelized, per-cell statistical models (GLMs and Cox proportional hazards) while adjusting for clinical covariates.
#
# In this tutorial, we will demonstrate the core workflow by investigating the tumor microenvironment (TME) of colorectal cancer (CRC). We will utilize the comprehensive ~4.2 million cell CRC atlas (Marteau et al., 2026) and project these discrete cellular profiles onto a combined bulk RNA-seq clinical cohort (TCGA, AC-ICAM, CMC-BSN). Ultimately, we will identify specific subpopulations driving patient survival, as well as those associated with critical tumor genotypes like Microsatellite Instability (MSI) and *KRAS* mutations.

# %% [markdown]
# ## Aligning Single-Cell and Bulk Modalities
#
# ```{eval-rst}
# .. list-table:: Analysis steps for modality alignment
#     :widths: 35 65
#     :header-rows: 1
#
#     - - shears / scanpy function
#       - biological objective
#     - - :func:`scanpy.pp.calculate_qc_metrics`
#       - Capture raw biological mRNA yields before filtering alters the data.
#     - - :func:`shears.deseq.filter_genes_deseq2`
#       - Robustly filter lowly expressed ambient noise in the bulk cohort.
#     - - :func:`shears.pp.calculate_batch_diversity`
#       - Flag biased, artificially sorted sequencing platforms that lack tissue diversity.
#     - - :func:`shears.pp.calculate_scaling_factors`
#       - Correct for intrinsic total mRNA yield differences between cell types.
#     - - :func:`shears.pp.scale_to_reference`
#       - Project bulk samples into the variance space of the single-cell reference.
# ```

# %%
# import the auto module and the standard text module
import tqdm.auto
from tqdm import tqdm as text_tqdm

# monkeypatch the auto version to permanently point to the text version
tqdm.auto.tqdm = text_tqdm

import os
import numpy as np
import pandas as pd
import scanpy as sc
import shears as sh
import decoupler as dc

# Configure scverse-standard logging and resource limits
sc.settings.n_jobs = 2
sh.settings.verbosity = 2

# Load the single-cell reference and bulk clinical cohorts
# adata_sc = sh.datasets.crc_atlas_marteau()
# adata_bulk = sh.datasets.crc_bulk_clinical()

adata_sc = sc.read_h5ad("/data/scratch/marteau/000_final_zenodo_upload/zenodo_upload_v2/shears_tutorial/adata_sc.h5ad")
adata_bulk = sc.read_h5ad("/data/scratch/marteau/000_final_zenodo_upload/zenodo_upload_v2/shears_tutorial/adata_bulk.h5ad")

# %%
os.environ["JOBLIB_TEMP_FOLDER"] = "/data/scratch/marteau/tmp/rm/"

# %% [markdown]
# :::{note} **Handling massive datasets with Joblib** # When computing models across millions of cells, `shears` memory-maps the data to disk. If your system's `/tmp` directory lacks capacity, redirect it to a high-capacity scratch partition by setting `os.environ["JOBLIB_TEMP_FOLDER"] = "/path/to/scratch/"` before running the pipeline. 
# :::
#
# ### Capturing total transcriptome size
#
# Later in the pipeline, we will explicitly correct for the fact that different cell types physically contain vastly different amounts of total RNA. To make this correction mathematically sound, we must record the absolute global transcriptome size of every single cell now, *before* we apply any gene filters.
#
# :::{important} 
# You must run {func}`scanpy.pp.calculate_qc_metrics` on the *unfiltered* reference. Subsetting genes to match the bulk data prior to this step fundamentally distorts the measurement. Rather than capturing the cell's true physical mRNA volume, you would merely be measuring the density of the overlapping markers.
# :::

# %%
# Capture raw biological mRNA yields ["total_counts"]
sc.pp.calculate_qc_metrics(
    adata_sc, percent_top=None, log1p=False, inplace=True,
)

# %% [markdown]
# ### Harmonizing library sizes and correcting length bias
#
# Bulk RNA-seq yields tens of millions of reads per sample, whereas a single cell captures only a few thousand Unique Molecular Identifiers (UMIs). Beyond this massive difference in sequencing depth, we must resolve a fundamental physical difference in how these technologies quantify transcripts:
#
# * **Bulk RNA-seq uses full-length protocols:** The entire mRNA transcript is fragmented and sequenced. Consequently, a long gene naturally shatters into more fragments—and generates proportionally more sequencing reads—than a short gene, even if their true biological expression is identical.
# * **Droplet scRNA-seq uses tag-based protocols:** Only the 3' or 5' end of the transcript is captured. Because each molecule is tagged with a UMI prior to amplification, we actively deduplicate PCR reads to count the absolute number of original molecules. This makes single-cell counts completely independent of gene length.
#
# If we ignore this difference, our algorithm will artificially inflate the expression of long genes in the bulk data. To safely force both modalities into a shared statistical space, we first strip the length bias out of the bulk data by converting raw counts to Transcripts Per Million (TPM). We then normalize both datasets to a fixed target sum.
#
# Finally, we apply a log(x+1) transformation. RNA expression is heavily right-skewed; without a log transformation, the linear Ridge solver would be entirely dominated by a few massively expressed housekeeping genes. The +1 is mathematically vital to ensure that true biological zeros remain exact zeros.

# %%
# Back up raw integer counts
adata_bulk.layers["counts"] = adata_bulk.X.copy()
adata_sc.layers["counts"] = adata_sc.X.copy()

# Compute bulk TPM to correct for gene length bias
length_kb = adata_bulk.var["Length"].to_numpy() / 1000
rpk = adata_bulk.X / length_kb
adata_bulk.layers["tpm"] = rpk / (rpk.sum(axis=1, keepdims=True) / 1e6)
adata_bulk.X = adata_bulk.layers["tpm"].copy()

# Harmonize library sizes and apply log transformation
sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.normalize_total(adata_bulk, target_sum=1e4)
sc.pp.log1p(adata_sc)
sc.pp.log1p(adata_bulk)

# %% [markdown]
# ### Aligning clinical contexts and harmonizing variance
#
# Before we can extract cell type signatures and deconvolute the bulk cohort, we must first filter our raw datasets and align the modalities through five critical steps:
#
# 1. **Establish a biological expression floor:** Bulk RNA-seq captures lowly expressed ambient RNA and degraded transcripts. We apply a robust statistical filter to discard genes that fail to show consistent expression across the cohort, preventing the algorithm from wasting statistical power on mathematical noise.
#
# 2. **Match clinical contexts:** The Marteau et al. (2026) CRC atlas encompasses the full spectrum of disease progression, from early polyps to advanced metastases. However, our target bulk cohort consists exclusively of treatment-naive primary tumors. If we leave polyps or metastases in the reference, the algorithm will erroneously distribute primary tumor reads across these incompatible disease states. We have to filter the single-cell atlas to mirror the clinical context of the bulk cohort.
#
# 3. **Provide a "Normal Sink":** We make one deliberate exception to our clinical matching. Bulk primary tumor samples inevitably encompass adjacent healthy tissue from the surgical margin. We explicitly retain healthy epithelial cells in our reference to absorb this contamination, otherwise, the algorithm would force these healthy reads onto malignant cancer cells.
#
# 4. **Harmonize the feature space:** Deconvolution algorithms select a highly optimized set of marker genes. By intersecting the matrices now, we drop single-cell markers that were discarded as ambient noise in the bulk tissue. This forces the engine to evaluate only robustly shared genes.
#
# 5. **Harmonize the variance space:** Finally, with our feature space perfectly matched, we fit a variance scaler exclusively on the single-cell reference and project the bulk data into that variance space. This preserves the true biological "brightness" of the markers, preventing the macroscopic bulk mixture from artificially distorting the variance.
#
# :::{important} Why we avoid Quantile Normalization
# Many older deconvolution tools default to Quantile Normalization to force bulk and single-cell data into the exact same statistical distribution. However, this approach is highly destructive to single-cell data. 
#
# Single-cell matrices are biologically sparse (>90% zeros). Quantile normalization destroys this sparsity by forcing true biological zeros to take on non-zero values from the bulk distribution, artificially hallucinating expression. Furthermore, while we rely on variance scaling to satisfy the L2 penalty of Ridge Regression, we explicitly do not mean-center the data. Standard scaling subtracts the mean, which would convert millions of structural zeros into negative floating-point numbers. Fitting a standard variance scaler on the pure single-cell reference without mean-centering perfectly aligns the macroscopic bulk mixture to our single-cell data while preserving structural zeros and the biological "brightness" of our marker genes.
# :::

# %%
# Dynamically calculate the bulk filtering threshold
n_samples_threshold = int(np.sqrt(adata_bulk.shape[0]))

# Establish a strict expression floor to discard ambient RNA
sh.deseq.filter_genes_deseq2(
    adata_bulk, min_samples=n_samples_threshold, min_counts=10, layer="counts"
)

# Strictly mirror the clinical state of the bulk cohort (treatment-naive, primary tumors)
# while explicitly retaining healthy epithelium as a Normal Sink for surgical margins
is_bulk_matched = adata_sc.obs.loc[
    lambda x: (
        (x["sample_type"] == "tumor") & (x["treatment_status_before_resection"] != "treated")
    ) | (
        (x["sample_type"] == "normal") & (x["cell_type"].isin(
            ["Crypt cell", "TA progenitor", "Colonocyte", "Colonocyte BEST4", "Goblet"]
        ))
    )
]

# Harmonize both objects to the ambient-filtered feature space
valid_bulk_genes = adata_bulk.var_names[adata_bulk.var["deseq2_keep"]]
common_genes = adata_sc.var_names.intersection(valid_bulk_genes)

adata_sc = adata_sc[is_bulk_matched.index, common_genes].copy()
adata_bulk = adata_bulk[:, common_genes].copy()

# Project bulk samples into the single-cell variance space
sh.pp.scale_to_reference(adata_sc, adata_bulk)

# %% [markdown]
# ### Deriving mRNA scaling factors from stable reference cells
#
# Cells inherently contain vastly different amounts of total RNA. A mature macrophage physically contains exponentially more transcripts than a resting T cell. If we do not correct for this biological disparity, deconvolution algorithms will structurally overestimate large cells and underestimate small cells in the bulk mixture.
#
# We must calculate intrinsic mRNA scaling factors on our finest cell type resolution to ensure we capture the precise biological capacity of distinct cell states, rather than blurring these differences across broad, heterogeneous lineages. To do this, `shears` uses a compositional data approach where it identifies a stable reference cell type to serve as the universal baseline (Factor = 1.0).

# %%
# Flag biased sequencing platforms lacking cellular diversity
sh.pp.calculate_batch_diversity(
    adata_sc, 
    groupby="cell_type", 
    batch_key="platform", 
    replicate_key="patient_id", 
    key_added="is_diverse_batch"
)

# Blacklist highly unstable, aneuploid, or phagocytic cells from reference calculation
unstable_references = [
    "Cancer Crypt-like", 
    "Cancer TA-like", 
    "Cancer Colonocyte-like",
    "Cancer Goblet-like", 
    "Cancer BEST4", 
    "Macrophage", 
]

# Calculate mRNA scaling factors on the finest resolution labels
sh.pp.calculate_scaling_factors(
    adata_sc, 
    groupby="cell_type", 
    batch_key="platform", 
    replicate_key="patient_id",
    subset_key="is_diverse_batch", 
    exclude_groups=unstable_references,
)

# %%
adata_sc.obs.groupby(["cell_type", "platform"], observed=True)["mRNA_scaling_factor"].first().unstack().head()

# %% [markdown]
# Looking at our single-cell reference dataset, the pipeline automatically identified the structurally stable **Non-classical Monocyte** as the optimal universal baseline (1.0). Relative to this baseline, a transcriptionally quiet **CD8+ T cell** sequenced on the 10x 3p platform receives a scaling factor of **0.43**. Conversely, a massive, highly active **Plasma IgA cell** receives a factor of **2.38**.
#
# Finding a true biological reference requires strict filtering:
#
# * **Excluding biased platforms:** Crucially, mRNA capture efficiency is a technical artifact of the sequencing chemistry, not the patient. If a platform lacks cellular diversity (e.g., early plate-based assays such as SMARTer C1 or scTrio-seq2 with exclusively epithelial cells), we cannot establish a universal structural baseline. We use `{func}~shears.pp.calculate_batch_diversity` to flag and exclude these homogenous platforms from the baseline calculation, allowing the algorithm to safely impute their factors from the diverse datasets later.
# * **Excluding unstable cell states:** We explicitly blacklist cancer cells and macrophages from being selected as the reference. Cancer cells suffer from massive copy number variations (CNVs) and aneuploidy, making their RNA yield highly erratic across patients, while macrophages are highly variable due to phagocytosis. The algorithm will automatically bypass these and search for a stable, diploid structural or immune cell (such as Monocytes, Fibroblasts, or Endothelial cells) to act as the platforms baseline.

# %% [markdown]
# ### Simplifying lineages to prevent weight smearing
#
# With our mRNA scaling factors safely derived from the fine cell type labels, we should now simplify the reference cell ontology to reflect the biological reality of bulk RNA-seq. 
#
# Bulk tissue is a macroscopic "smoothie" of millions of cells. The transcriptional difference between closely related sibling states, like a "Naive B cell" and a "Memory B cell", is incredibly subtle. In a complex bulk mixture, this microscopic sub-lineage signal is completely drowned out by technical noise. 
#
# This biological noise creates a critical mathematical vulnerability. The core deconvolution engine of `shears` (`{func}~shears.pp.cell_weights`) relies on non-negative ridge regression (L2 regularization). If we force the Ridge solver to choose between nearly identical sibling states, it suffers from severe multicollinearity. Rather than finding a true biological footprint, the L2 penalty indiscriminately 'smears' the predicted cell fractions across these highly correlated states, obscuring the true biological signal.
#
# Furthermore, immune cells tend to traffic together in broad, coordinated networks (e.g., globally "hot" inflamed tumors vs. "cold" immune deserts). Coarse-graining our labels not only stabilizes the Ridge regression, but it massively increases our statistical power for downstream clinical modeling by grouping scarce cells into robust, stable functional modules.
#
# To stabilize the model and increase our downstream statistical power, we perform two types of merging:
#
# * **Lineage merging:** Consolidating developmental states that share too much transcriptomic machinery to be distinguished in bulk (e.g., merging "B cell naive", "B cell memory", and "GC B cell" into a unified "B cell" compartment).
# * **Functional merging:** Collapsing disease-specific spatial phenotypes (e.g., merging "Cancer Crypt-like" and "Cancer Colonocyte-like") into a universal "Cancer cell" state. This forces the algorithm to confidently capture the overall macroscopic tumor burden rather than getting confused by microscopic spatial variations.

# %%
# Merge fine-grained labels to resolve multicollinearity
adata_sc.obs["cell_type"] = (
    adata_sc.obs["cell_type"]
    .astype(str)
    .replace(
        {
            "B cell naive": "B cell",
            "B cell memory": "B cell",
            "GC B cell": "B cell",
            "Plasmablast": "Plasma cell",
            "Plasma IgA": "Plasma cell",
            "Plasma IgG": "Plasma cell",
            "Plasma IgM": "Plasma cell",
            "T cell regulatory": "T cell CD4",
            "Monocyte classical": "Monocyte",
            "Monocyte non-classical": "Monocyte",
            "DC3": "cDC2",
            "DC mature": "cDC2",
            "Fibroblast S1": "Fibroblast",
            "Fibroblast S2": "Fibroblast",
            "Fibroblast S3": "Fibroblast",
            "Endothelial venous": "Endothelial cell",
            "Endothelial arterial": "Endothelial cell",
            "Endothelial lymphatic": "Endothelial cell",
            "Cancer Crypt-like": "Cancer cell",
            "Cancer TA-like": "Cancer cell",
            "Cancer Colonocyte-like": "Cancer cell",
            "Cancer Goblet-like": "Cancer cell",
            "Cancer BEST4": "Cancer cell",
            "TA progenitor": "Epithelial progenitor",
            "Crypt cell": "Epithelial progenitor",
        }
    )
    .astype("category")
)

# %% [markdown]
# ## Resolving Collinearity to Extract Robust Signatures
#
# With our reference cell ontology simplified, we can now build the mathematical feature matrix (the gene signatures) that will drive the deconvolution. 
#
# We cannot rely on standard Highly Variable Genes (HVGs). Standard HVG algorithms penalize genes with low global variance, meaning they will actively delete rare "on/off" genes—the exact genes that act as the holy grail for identifying rare cell types in a bulk mixture. Furthermore, because non-negative Ridge Regression is an inherently additive model, we must strictly isolate positive markers (the "bright lightbulbs" of a cell type). Downregulated genes only confuse the algorithm, as you cannot reliably deduce the presence of a cell from the *absence* of a signal in a complex mixture.
#
# ```{eval-rst}
# .. list-table:: Analysis steps for signature engineering
#     :widths: 35 65
#     :header-rows: 1
#
#     - - shears / decoupler function
#       - biological objective
#     - - :func:`decoupler.pp.pseudobulk`
#       - Aggregate sparse single cells into robust biological replicates.
#     - - :func:`shears.deseq.rank_genes_sum2zero`
#       - Identify true lineage markers while algebraically filtering out sparsity artifacts.
#     - - :func:`shears.tl.flag_stable_features`
#       - Establish a bulk noise floor to prevent rare single-cell markers from being buried.
#     - - :func:`shears.tl.extract_signatures`
#       - Isolate optimal marker genes using Bipartite matching to prevent gene-stealing.
#     - - :func:`shears.tl.prune_signatures`
#       - Remove single-cell markers driven by off-target "Loud Neighbors" in the bulk mixture.
# ```
#
# ### Pseudobulking the single-cell atlas
#
# To perform robust differential expression and extract these marker genes, we first aggregate our sparse single-cell profiles into pseudobulk samples, grouping by patient and cell type. This provides us with stable biological replicates and true integer counts.

# %%
# Pseudobulk the single-cell atlas by patient and cell type
pb = dc.pp.pseudobulk(
    adata=adata_sc,
    sample_col="patient_id",
    groups_col="cell_type",
    layer="counts",
    mode="sum",
)

# Carry over necessary patient-level metadata
pb.obs["dataset"] = pb.obs["patient_id"].map(
    adata_sc.obs.groupby("patient_id").first()["dataset"]
)

# Filter out low-quality pseudobulk profiles
dc.pp.filter_samples(pb, min_cells=20, min_counts=1000)

# %% [markdown]
# ### Defining structural anchors and functional domains
#
# To extract distinct markers for closely related lineages without re-introducing collinearity, we deploy a hierarchical dual-mapping strategy. We decouple our macroscopic **Structural Anchors** from our microscopic **Functional Domains** (Splitters).
#
# **1. The Structural Anchors (Macro-Lineages)**
# Anchors provide the generic, foundational footprints (e.g., *EPCAM* for Epithelium, *CD3E* for T cells) that lock the Ridge solver into the correct macroscopic biological compartment. This mapping strictly follows standard developmental biology. Notice that we keep `Malignant` separate so the tumor defines its own unique anchor.
#
# **2. The Functional Domains (Splitters)**
# Splitters act as the highly specific barcodes for a cell state. Standard "top-N" marker selection fails here because sibling cell types share highly significant markers. By grouping overlapping states into shared functional domains, we force the downstream extraction algorithm to globally arbitrate contested genes, ensuring mutually exclusive fingerprints.
#
# We define four major functional domains to resolve collinearity:
# * **Pan-Lymphocyte:** Forces T cells and NK cells to resolve specific lineage markers (*CD8A* vs. *NCAM1*) rather than confounding the model with shared cytotoxic machinery (*GZMB*, *PRF1*).
# * **Pan-Myeloid & APC:** The highest collinearity risk in the tissue. This massive grouping prevents B cells, DCs, and Macrophages from cross-contaminating the MHC-II locus, and forces specialized granulocytes to rely on unique enzymes rather than generic inflammatory markers.
# * **Pan-Epithelial:** Prevents malignant cancer cells from confounding the model with generic epithelial structural genes (*KRT8*/*KRT18*)
# * **Pan-Stromal:** Prevents the sharing of generic mesenchymal and matrix adhesion components across endothelial cells and fibroblasts.

# %%
# Define macroscopic structural anchors
anchor_mapping = {
    "T lymphocyte": ["T cell CD8", "T cell CD4", "T cell gd"],
    "B lymphocyte": ["B cell", "Plasma cell"],
    "Innate lymphocyte": ["NK cell", "ILC"],
    "Myeloid": ["Monocyte", "Macrophage", "cDC1", "cDC2", "pDC"],
    "Granulocyte": ["Neutrophil", "Mast cell", "Eosinophil", "Granulocyte progenitor"],
    "Malignant": ["Cancer cell"], 
    "Epithelial cell": [
        "Enteroendocrine", "Tuft", "Colonocyte", "Colonocyte BEST4", 
        "Goblet", "Epithelial progenitor"
    ],
    "Stromal cell": ["Endothelial cell", "Fibroblast", "Pericyte"],
    "Glial cell": ["Schwann cell"],
}

# Define functional domains to resolve multicollinearity
splitter_mapping = {
    "Pan-Lymphocyte": ["T cell CD8", "T cell CD4", "T cell gd", "NK cell", "ILC"],
    "Pan-Myeloid & APC": [
        "B cell", "Plasma cell", "Monocyte", "Macrophage",
        "cDC1", "cDC2", "pDC", "Neutrophil", "Mast cell",
        "Eosinophil", "Granulocyte progenitor"
    ],
    "Pan-Epithelial": [
        "Cancer cell", "Enteroendocrine", "Tuft", "Epithelial progenitor",
        "Colonocyte", "Colonocyte BEST4", "Goblet"
    ],
    "Pan-Stromal": ["Endothelial cell", "Fibroblast", "Pericyte", "Schwann cell"],
}

# Apply mappings to the pseudobulk object
fine_to_anchor = {f: c for c, f_list in anchor_mapping.items() for f in f_list}
fine_to_splitter = {f: c for c, f_list in splitter_mapping.items() for f in f_list}

pb.obs["anchor_coarse"] = pb.obs["cell_type"].map(fine_to_anchor).astype("category")
pb.obs["splitter_coarse"] = pb.obs["cell_type"].map(fine_to_splitter).astype("category")

# %% [markdown]
# :::{note} **What if a lineage only has one cell type?**
# Notice that `Schwann cell` is isolated in its own "Glial cell" category. If a lineage has no siblings, it does not require a Splitter. The broad Anchor genes mathematically suffice to define its cell state.
# :::
#
# ### Ranking genes using sum-to-zero contrasts
#
# Standard differential tests frequently highlight pan-markers (like *CD45*), which hold zero value for deconvolution because they cannot distinguish between specific immune cells. We utilize a "sum-to-zero" contrast via PyDESeq2, actively penalizing generic genes and isolating true, lineage-defining markers. 
#
# :::{dropdown} Technical Details: The Cholesky Shield against Sparsity Artifacts
# When transitioning to single-cell pseudobulk, a mathematical hazard known as "complete separation" occurs. If a gene has exactly zero counts across an entire biological factor (e.g., a specific patient), the design matrix becomes perfectly confounded. Because $\lim_{\text{background counts} \to 0} \text{LogFoldChange} = \infty$, traditional GLM optimizers hallucinate an artificially massive fold-change with absolute mathematical confidence (a "perfect" p-value of $0.0$).
#
# Because stricter p-value thresholding actually *enriches* for this mathematical noise, `shears` relies on linear algebra rather than statistics to solve the problem. By enabling `filter_rank_deficient_genes=True`, the algorithm performs a Cholesky decomposition ($X^T X = L L^T$) to mathematically evaluate the full column rank of the design matrix *before* the GLM is initialized, dropping rank-deficient genes from the "sum-to-zero" contrast.
# :::

# %%
# Retain robustly represented cell types
valid_cell_types = pb.obs["cell_type"].value_counts()[lambda x: x > 3].index
pb = pb[pb.obs["cell_type"].isin(valid_cell_types)].copy()

# Rank Structural Anchors across the entire dataset
sh.deseq.rank_genes_sum2zero(
    pb,
    groupby="anchor_coarse",
    design="~dataset + anchor_coarse",
    rank_filter_method="cholesky",
    gene_symbols="GeneSymbol",
    key_added="rank_genes_sum2zero_anchors",
    n_jobs=cpus,
)

# Target the Splitter column so cells compete strictly within their specific functional domain
sh.deseq.rank_genes_sum2zero(
    pb,
    groupby="cell_type",
    design="~dataset + cell_type",
    subset_key="splitter_coarse",
    rank_filter_method="cholesky",
    gene_symbols="GeneSymbol",
    key_added="rank_genes_sum2zero_splitters",
    n_jobs=cpus,
)

# %%
sh.get.rank_genes_sum2zero_df(
    pb, subset_groups="Neutrophil", key="rank_genes_sum2zero_splitters"
)

# %% [markdown]
# ### Extracting mutually exclusive markers
#
# Before we extract our markers, we must establish a bulk noise floor using `{func}~shears.tl.flag_stable_features`. We utilize an optimal intermediate threshold (`min_bulk_quantile=0.15`). 
#
# In bulk RNA-seq, the bottom 10-15% of the transcriptome is largely ambient noise or technical dropouts. Dropping the floor to 0.05 contaminates the feature matrix with ambient zeros. However, raising it to 0.30 inadvertently deletes the faint but highly specific marker genes of rare populations (like CD8+ T cells) before the math even begins. A threshold of 0.15 perfectly slices off the ambient static while preserving true biological signals.
#
# Next, we parameterize our extractions for the two marker classes:
# * **Anchors require a "Pan-Lineage Footprint":** Lineage markers are inherently shared. We utilize `method="standard"` with a strict physical noise floor (`basemean_quantile=0.75`), forcing the selection of universally bright genes.
# * **Splitters require a "State-Specific Barcode":** We utilize `method="bipartite"` to prevent closely related lineages from monopolizing shared markers. Under the hood, this utilizes the Hungarian algorithm to frame feature selection as a global assignment problem. It acts as an impartial judge across the functional domain, ensuring highly similar lineages receive mutually exclusive barcodes rather than misallocating contested activation markers.
#
# :::{important} The Anchor Depletion Trap & Mathematical Equalization
# Notice that we cast a massive net for Anchors (`n_top_features=100`) but strictly cap our Splitters (`n_top_features=35`). This is a mechanical necessity. 
#
# The bipartite algorithm enforces strict exclusivity. If we demanded 100 splitters per cell type, the algorithm would be forced to select weakly expressed background genes, locking thousands of generic lineage genes into an exclusion list. Desperate to hit its quota, it would inadvertently filter out the parent lineage anchors, depriving the model of foundational structural markers. Furthermore, capping the signature length acts as a mathematical equalizer, preventing transcript-heavy cells (like Macrophages) from generating massive signatures that mathematically overshadow rare cells in the downstream L2 solver.
# :::

# %%
sh.tl.flag_stable_features(
    adata_bulk=adata_bulk,
    adata_pb=pb,
    gene_symbols="GeneSymbol",
    layer_bulk="ref_scaled",
    key_added="stable_features",
)

# Extract Anchors via Standard Footprint Extraction
sh.tl.extract_signatures(
    adata_pb=pb,
    source_key="rank_genes_sum2zero_anchors",
    n_top_features=100,  # <- oversample!
    padj_threshold=0.05,
    min_lfc=1.0,
    basemean_quantile=0.75,
    method="standard",
    stable_features_key="stable_features",
    key_added="shears_anchors",
)

# Extract Splitters via Bipartite Barcoding
sh.tl.extract_signatures(
    adata_pb=pb,
    source_key="rank_genes_sum2zero_splitters",
    n_top_features=50,  # <- extract deep, assign shallow
    padj_threshold=0.05,
    min_lfc=1,
    basemean_quantile=0.25,
    method="bipartite",
    stable_features_key="stable_features",
    key_added="shears_splitters",
)

# Merge into a safe 1:7 (Anchor:Splitter) ratio
sh.tl.merge_signatures(
    adata_pb=pb,
    coarse_key="shears_anchors",
    fine_key="shears_splitters",
    n_anchors=5,
    n_splitters=25,
)

# %% [markdown]
# ### Pruning the "Loud Neighbor" Effect
#
# A gene may serve as a perfect, unique marker for a rare pDC in a healthy single-cell reference. However, in the chaotic bulk tumor environment, that exact transcript might be strongly upregulated by inflamed endothelial cells. 
#
# {func}`~shears.tl.prune_signatures` evaluates the signatures against the *actual bulk tissue*. If a marker fails to move synchronously with the rest of its cell-type module in the bulk matrix, it represents a "Loud Neighbor" and is safely discarded. This explicitly prevents massive structural compartments from stealing the collinear activation signals of smaller immune populations.

# %%
sh.tl.prune_signatures(
    adata_bulk=adata_bulk,
    adata_pb=pb,
    signature_key="shears_signatures",
    min_corr=0.15,
    layer="ref_scaled",
)

# Retrieve finalized genes and subset datasets
signatures = sh.get.signatures(adata_bulk, key="shears_signatures")
final_genes = list({gene for gene_list in signatures.values() for gene in gene_list})

# %%
for ct, genes in sorted(signatures.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"{ct}: {len(genes)}")

# %%
len(final_genes)

# %%
splitters = sh.get.signatures(pb, key="shears_splitters")
anchors = sh.get.signatures(pb, key="shears_anchors")

# Splitters are already mapped directly to the fine cell types
splitter_pairs = {(ct, g) for ct, genes in splitters.items() for g in genes}

# For anchors, we unpack the list of fine cell types belonging to each coarse group
anchor_pairs = {
    (fine_ct, gene)
    for coarse_ct, fine_list in anchor_mapping.items()
    for fine_ct in fine_list
    for gene in anchors.get(coarse_ct, [])
}

df_signatures = (
    sh.get.rank_genes_sum2zero_df(
        pb, key="rank_genes_sum2zero_splitters", signatures_key="shears_signatures"
    )
    .assign(
        marker_type=lambda df_in: np.select(
            condlist=[
                pd.MultiIndex.from_frame(df_in[["cell_type", "var_names"]]).isin(
                    splitter_pairs
                ),
                pd.MultiIndex.from_frame(df_in[["cell_type", "var_names"]]).isin(
                    anchor_pairs
                ),
            ],
            choicelist=["Splitter", "Anchor"],
            default="Not Selected",
        )
    )
    .query("marker_type != 'Not Selected'")
    .reset_index(drop=True)
)

# %%
df_signatures[["cell_type", "GeneSymbol", "marker_type"]]

# %%
sh.tl.score_signature_detectability(adata_bulk)

# %%
adata_sc = adata_sc[adata_sc.obs["cell_type"].isin(signatures.keys()), adata_sc.var_names.isin(final_genes)].copy()
adata_bulk = adata_bulk[:, adata_sc.var_names].copy()

# %% [markdown]
# ## Deconvoluting the Bulk Tissue
#
# Before computing the final cell fractions, we must optimize the single-cell reference matrix to balance the global L2 regularization penalty of our downstream Ridge solver. 
#
# If we feed all 2 million cells directly into the solver, hyper-abundant populations (like malignant cancer cells) will overwhelm the matrix with redundant biological clones. This massive class imbalance forces the solver to minimize loss by fitting the abundant clones perfectly, mathematically overshadowing the subtle signals of globally rare populations.
#
# To prevent this, we utilize `{func}~shears.pp.downsample_reference` to apply a hybrid stratified downsampling strategy:
# 1. **Noise compression:** It drops minor replicates (e.g., `min_cells=20`) that lack the statistical degrees of freedom to establish a reliable intra-patient biological profile.
# 2. **Rare population preservation:** It explicitly protects globally rare cells (`global_floor=5000`) from downsampling, ensuring their mathematical footprint remains robust.
# 3. **Abundance capping:** It randomly downsamples hyper-abundant cells to a strict ceiling (`batch_cap=250`) per patient, preserving true intra-sample variance (e.g., activation gradients) while discarding mathematically redundant clones.
#
# :::{important} The Post-Hoc Fallacy and The Anchor Clamp
# It is tempting to skip upstream lineage simplification, feed the solver highly collinear sibling states, and simply sum their predicted weights together post-hoc. **This is mathematically invalid.** # 
# Post-hoc addition cannot resurrect signals suppressed by Ridge regression. If distinct siblings share an effector signal but possess distinct structural anchors, the Ridge solver cannot assign a large weight without drastically over-predicting the unique anchors. To avoid the massive $||Y - Xw||_2^2$ penalty, the distinct anchors act as a mathematical "clamp", forcing the solver to suppress the weights of *both* siblings and leaving the shared signal un-fitted. We must merge highly correlated states *before* extraction to dilute these conflicting anchors.
# :::
#
# :::{note} Preserving the Isotropic Assumption
# We deliberately compute the Ridge regression *before* applying the biological mRNA scaling factors. The L2 penalty ($\alpha||w||_2^2$) assumes all input features are on an equal playing field. By running the solver in a purely normalized mathematical space first, we preserve this isotropic assumption. We then safely apply the biological scaling factors post-hoc.
# :::

# %%
# Optimize the reference matrix to balance the L2 penalty
sh.pp.downsample_reference(adata_sc, groupby="cell_type", batch_key="patient_id")

# %%
# Compute cell fractions via Ridge regression
sh.pp.cell_weights(
    adata_sc=adata_sc,
    adata_bulk=adata_bulk,
    subset_key="shears_downsampled",
    alpha_callback=lambda ad: 10 * ad.shape[0],
    layer_sc="ref_scaled",
    layer_bulk="ref_scaled",
    key_added="cell_weights",
    n_jobs=-1,
)

# %% [markdown]
# ### Aggregation and biological bias correction
#
# With our raw mathematical weights calculated by the Ridge solver, we can now aggregate them to the patient level and correct for the inherent biological disparity in cell sizes. 
#
# We utilize `{func}~shears.tl.aggregate_obsm_group` to collapse the individual single-cell weights into comprehensive cell-type profiles per patient. Crucially, it is during this aggregation that we apply the `mRNA_scaling_factor`. By multiplying the raw weights by this factor, we transition from purely mathematical "transcriptional contributions" to true, biologically representative "cell counts." 
#
# :::{important} Order of Operations 
# The biological bias correction must happen *now*. It drastically alters the relative magnitudes of the cells (e.g., inflating quiet T cells and suppressing massive Macrophages). If we applied global mathematical scaling *before* this step, this biological multiplication would blow the values right back out of proportion and crash our downstream statistical solvers.
# :::

# %%
# Strip out unmapped cells and apply mRNA scaling factors
sh.tl.filter_unmapped_cells(adata_sc)

sh.tl.transfer_weights(
    adata_sc,
    adata_bulk,
    groupby="cell_type",
    batch_key="patient_id",
    subset_key="nonzero_bulk_weight",
    scaling_factor_col="mRNA_scaling_factor",
    min_cells=20,
    agg_type="sum",
)

# Isolate the normal surgical margin to explicitly renormalize the true TME
adata_sc.obs["cell_type"] = (
    adata_sc.obs["cell_type"]
    .astype(str)
    .replace(
        {
            "Colonocyte": "Epithelial normal",
            "Epithelial progenitor": "Epithelial normal",
            "Colonocyte BEST4": "Epithelial normal",
            "Goblet": "Epithelial normal",
        }
    )
    .astype("category")
)

sh.tl.compartment_fraction(
    adata_bulk,
    compartment_groups=["Epithelial normal"],
    key_added="surgical_margin_fraction",
)

# %%

# %% [markdown]
# ## Modeling Clinical Outcomes at Single-Cell Resolution
#
# ```{eval-rst}
# .. list-table:: Analysis steps for clinical modeling
#     :widths: 35 65
#     :header-rows: 1
#
#     - - shears function
#       - biological objective
#     - - :func:`shears.tl.shears_glm`
#       - Test cell abundance against categorical phenotypes, adjusting for confounders.
#     - - :func:`shears.tl.shears_cox`
#       - Test cell abundance against time-to-event outcomes (Overall Survival).
#     - - :func:`shears.tl.differential_composition`
#       - Aggregate cell-level signal while filtering out noise using the Wald statistic.
# ```
#
# We are finally ready to test our deconvoluted cell fractions against clinical outcomes (e.g., Microsatellite Instability). However, we must address two final statistical hazards before running our Generalized Linear Models (GLMs).
#
# **1. Regressing out the "Normal Sink"**
# Our samples are bulk tissue blocks, meaning they inevitably contain adjacent normal margin. If a sample is 80% normal colon, the immune cells are artificially squeezed into the remaining 20%. If we simply correlate CD8+ T cells with MSI status, we aren't measuring immune infiltration; we are measuring surgical margin dilution! 
#
# We *cannot* mathematically delete the normal weights from the matrix, as this destroys the linear additivity assumptions of the GLM. Instead, we calculate the `surgical_margin_fraction` and pass it directly into the GLM design formula as a covariate. The model will realize, *"The CD8+ T-cell weight is low here, but that is just because the surgical margin is 90%. I won't penalize the tumor's immune score for that."*
#
# **2. Preventing the Hauck-Donner Effect**
# The raw cell weights are often microscopic (e.g., 0.000001). If we feed these into `statsmodels` alongside normal covariates (like `age_scaled`), the optimization solver panics attempting to calculate the gradients and shoots the coefficients to infinity (The Hauck-Donner effect). By enabling `scale_weights=True` in the GLM, `shears` dynamically applies Max-Abs scaling. It divides the entire weights matrix by its global maximum, perfectly bounding the predictors between 0.0 and 1.0 without altering the underlying biological p-values.
#
# ### Adjusting for Confounders and Protecting the FDR
#
# Bulk cohorts are noisy. A cell type might seemingly correlate with death when, in reality, it is simply more abundant in older patients. By defining explicit covariates—including the `surgical_margin_fraction`—the model mathematically regresses out demographic confounders and epithelial contamination.
#
# Furthermore, we must protect Statistical Power via the **"Max in n Patients" rule**. If we feed zero-inflated rare cell types into a GLM, they punish our False Discovery Rate (FDR) multiple-testing correction, destroying the statistical power of truly abundant cells.

# %%
# Exclude bulk samples with nans in columns of interest
exclude = adata_bulk[
    (adata_bulk.obs["microsatellite_status"].isna())
    | (adata_bulk.obs["tumor_stage"].isna())
    | (adata_bulk.obs["cancer_type"].isna())
].obs_names
bulk_metadata = adata_bulk[~adata_bulk.obs_names.isin(exclude)].obs
bulk = adata_bulk[~adata_bulk.obs_names.isin(exclude)].copy()

bulk.obs["microsatellite_status"] = pd.Categorical(
    bulk.obs["microsatellite_status"], categories=["MSS", "MSI"]
)

# Define the confounder formula
covariate_str = (
    "C(stage, Treatment(reference='early'))"
    " + C(sex, Treatment(reference='male'))"
    " + age_scaled"
    "+ C(project, Treatment(reference='TCGA'))"
    "+ surgical_margin_fraction"
)

sh.tl.shears_glm(
    adata_sc,
    bulk,
    dep_var="microsatellite_status",
    key_added="shears_glm_microsatellite_status",
    covariate_str=covariate_str,
    n_jobs=-1,
)

# %%
# Aggregate signal utilizing the Wald Statistic
sh.tl.differential_composition(
    adata_sc=adata_sc,
    model_key="shears_glm_KRAS_mutation",
    groupby="cell_type",
    batch_key="patient_id",
    scaling_key="mRNA_scaling_factor",
)

# %%
sh.pl.differential_composition(adata_sc, exclude_ns=True)

# %% [markdown]
# ### Fitting Statistical Models 
#
# `shears` computes massively parallelized Generalized Linear Models (GLMs) and Cox models for *every single cell* against the bulk cohort, preserving the subtle transcriptomic variance of the atlas.
#
# :::{dropdown} Technical Details: HPC Orchestration and the Hessian Rescue
# Computing models across millions of cells requires extreme High-Performance Computing (HPC) engineering. `shears` pre-compiles design matrices in the main thread and allocates C-contiguous NumPy memory, dropping workers into pure Cython linear algebra to bypass Python's Garbage Collector. Furthermore, because deconvolution weights are often microscopic fractions, standard Newton-Raphson optimizers crash due to Hessian singularities. `shears` dynamically divides cell weights by their global maximum. This scales inputs to safe bounds while perfectly preserving the Signal-to-Noise ratio and biological interpretation. 
# :::

# %%
exclude = adata_bulk[
    (adata_bulk.obs["OS_status"].isna())
    | (adata_bulk.obs["OS_time"].isna())
    | (adata_bulk.obs["tumor_stage"].isna())
].obs_names
bulk_metadata = adata_bulk[~adata_bulk.obs_names.isin(exclude)].obs

bulk = adata_bulk[~adata_bulk.obs_names.isin(exclude)].copy()

covariate_str = (
    "+ C(stage, Treatment(reference='early'))"
    " + C(sex, Treatment(reference='male'))"
    " + age_scaled"
    "+ surgical_margin_fraction"
)

sh.tl.shears_cox(
    adata_sc,
    bulk,
    duration_col="OS_time",
    event_col="OS_status",
    covariate_str=covariate_str,
    n_jobs=-1,
    init_kwargs={"strata": ["project"]},
)

# %%
# Aggregate signal utilizing the Wald Statistic
sh.tl.differential_composition(
    adata_sc=adata_sc,
    model_key="shears_cox",
    groupby="cell_type",
    batch_key="patient_id",
    scaling_key="mRNA_scaling_factor",
)

# %%
sh.pl.differential_composition(adata_sc, exclude_ns=True)

# %% [markdown]
# ## Visualizing Clinical Associations and Reproducibility
#
# Single-cell data remains highly sparse. We clean up these individual fits using the **Wald Statistic**, acting as a strict signal-to-noise ratio dividing the biological effect size by its mathematical uncertainty. 

# %% [markdown]
# ### Interpreting the Output
#
# The Differential Composition Boxplot translates millions of single-cell fits into a single, intuitive summary of your phenotype-associated subpopulations.
#
# * **The X-Axis (Signal-to-Noise):** Positive shifts indicate confident enrichment (e.g., worse survival); negative shifts indicate confident depletion.
# * **The Color (FDR Significance):** Colored boxes have passed strict False Discovery Rate correction. Gray boxes ("n.s.") indicate that mathematical uncertainty outweighs any potential biological effect.
# * **The Box Spread (Biological Reproducibility):** Overlaid dots represent independent single-cell biological replicates (the human donors). A tight box confirms the clinical association as a universal biological truth, rather than a batch artifact from a single donor.
# * **The Convergence Rate (e.g., `CD8_Teff [98%]`):** This measures Mathematical Trust. It tracks exactly how many individual cells of that lineage successfully fit the clinical model. Do not trust a significant p-value if the convergence is below 25%—the algorithm simply did not have enough stable data to form a reproducible footprint.
#
# :::{note} **Understanding Coefficient Shrinkage** # In high-dimensional biology, models often assign massive coefficients (e.g., -10) to rare cells to "force" fits on noisy data. Deep inside the statistical engine, `shears` utilizes Winsorization to clip extreme technical outliers. If an association appears with a modest score (e.g., -2), this represents a mathematically stable, highly reproducible finding prioritized to withstand validation in external cohorts. 
# :::

# %%
import session_info

session_info.show()
