# Datasets

Functions for fetching external data resources (TF collections, gene annotations, example datasets).

## TF Collections

```{eval-rst}
.. autofunction:: deepscenic.fetch_tf_collection
```

## Gene Annotations

```{eval-rst}
.. autofunction:: deepscenic.fetch_gene_annotation
```

## Example Datasets

```{eval-rst}
.. autofunction:: deepscenic.fetch_mm_lines
```

## Cache Management

```{eval-rst}
.. autofunction:: deepscenic.get_cache_info
.. autofunction:: deepscenic.clear_cache
```

## Usage

```python
import deepscenic as ds

# Fetch mouse TF list from SCENIC+ resources
tfs = ds.fetch_tf_collection(species="mouse")
print(f"Found {len(tfs)} transcription factors")

# Fetch gene annotations from Ensembl
annot, chromsizes = ds.fetch_gene_annotation(
    species="mmusculus",
    biomart_host="http://nov2020.archive.ensembl.org/",
)

# Fetch the MM_lines melanoma dataset used in the tutorials (GEO GSE210745)
adata_rna, adata_atac, metadata = ds.fetch_mm_lines()

# Check cache status
ds.get_cache_info()

# Clear cached data
ds.clear_cache()
```
