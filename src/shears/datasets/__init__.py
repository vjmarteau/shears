"""Demo datasets."""


def tcga(cohort, layers=("TPM", "counts")):
    """Download TCGA data from firebrowse and process it into an anndata object."""
    # TODO how easy is metadata?
    raise NotImplementedError


def tcga_nsclc():
    """TCGA NSCLC cohort with curated metadata."""
    # TODO this one should be easy, we have it already.
    raise NotImplementedError


def luca_30k():
    """A subset of the lung cancer atlas (LuCA) including 3 datasets with 10 patients each, subsampled to 10k cells each."""
    # TODO need to create this and host on GitHub.
    raise NotImplementedError
