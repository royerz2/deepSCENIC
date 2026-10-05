"""Tests for dataset fetching utilities."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pooch
import pytest

import deepscenic as ds
from deepscenic._datasets import MM_LINES_FILES, TF_COLLECTION_CHECKSUMS, TF_COLLECTION_URLS, _read_tsv_matrix


class TestFetchTFCollection:
    """Tests for fetch_tf_collection function."""

    def test_invalid_species(self):
        """Test error on invalid species."""
        with pytest.raises(ValueError, match="Unknown species"):
            ds.fetch_tf_collection(species="invalid")  # type: ignore[arg-type]

    def test_species_urls_have_hash(self):
        """Test that all expected species have URLs defined."""
        for key in TF_COLLECTION_URLS:
            assert key in TF_COLLECTION_CHECKSUMS, f"URL key {key} has no matching checksum"

    @patch("pooch.retrieve")
    def test_fetch_returns_list(self, mock_retrieve, tmp_path):
        """Test that fetch returns a list of TF names."""
        # Create mock TF file
        tf_file = tmp_path / "allTFs_mouse.txt"
        tf_file.write_text("Adnp\nAebp1\nAebp2\nAhr\n")

        mock_retrieve.return_value = str(tf_file)

        tfs = ds.fetch_tf_collection(species="mouse")

        assert isinstance(tfs, list)
        assert len(tfs) == 4
        assert tfs[0] == "Adnp"
        assert tfs[3] == "Ahr"


class TestFetchGeneAnnotation:
    """Tests for fetch_gene_annotation function."""

    @patch("deepscenic._datasets._fetch_from_biomart")
    def test_invalid_species_raises(self, mock_fetch):
        """Test error on invalid species."""
        # Setup mock to return empty dataset list
        mock_fetch.side_effect = ValueError("Dataset 'invalid_species_xyz_gene_ensembl' not found.")

        with pytest.raises(ValueError, match="not found"):
            ds.fetch_gene_annotation(species="invalid_species_xyz", force_download=True)

    @patch("deepscenic._datasets._fetch_from_biomart")
    def test_fetch_returns_dataframe(self, mock_fetch, tmp_path, monkeypatch):
        """Test that fetch returns properly formatted DataFrame."""
        # Redirect cache to temp directory
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        # Mock query response
        mock_annot = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr1", "chr2"],
                "Start": [1000, 2000, 3000],
                "End": [1500, 2500, 3500],
                "Strand": ["+", "-", "+"],
                "Transcription_Start_Site": [1000, 2500, 3000],
                "Transcript_type": ["protein_coding"] * 3,
            },
            index=pd.Index(["Gene1", "Gene2", "Gene3"], name="Gene"),
        )
        mock_fetch.return_value = (mock_annot, None)

        annot, _ = ds.fetch_gene_annotation(species="mmusculus")

        # Verify result structure
        assert isinstance(annot, pd.DataFrame)
        assert annot.index.name == "Gene"
        assert "Chromosome" in annot.columns
        assert "Transcription_Start_Site" in annot.columns
        assert len(annot) == 3

        # Check strand values preserved
        assert annot.loc["Gene1", "Strand"] == "+"
        assert annot.loc["Gene2", "Strand"] == "-"


class TestGeneAnnotationCaching:
    """Tests for parquet caching of gene annotations."""

    def test_cache_created(self, tmp_path, monkeypatch):
        """Test that parquet cache is created after fetch."""
        # Redirect cache to temp directory
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        # Mock the biomart fetch to avoid network
        with patch("deepscenic._datasets._fetch_from_biomart") as mock:
            mock_annot = pd.DataFrame(
                {"col": [1, 2]},
                index=pd.Index(["A", "B"], name="Gene"),
            )
            mock.return_value = (mock_annot, None)

            ds.fetch_gene_annotation()

        # Check cache file exists
        cache_files = list(tmp_path.glob("gene_annot_*.parquet"))
        assert len(cache_files) == 1

    def test_cache_used_on_second_call(self, tmp_path, monkeypatch):
        """Test that second call uses cache without fetching."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        # First call - creates cache
        with patch("deepscenic._datasets._fetch_from_biomart") as mock:
            mock_annot = pd.DataFrame(
                {"col": [1, 2]},
                index=pd.Index(["A", "B"], name="Gene"),
            )
            mock.return_value = (mock_annot, None)
            ds.fetch_gene_annotation()
            assert mock.call_count == 1

        # Second call - should use cache, not fetch
        with patch("deepscenic._datasets._fetch_from_biomart") as mock:
            annot, _ = ds.fetch_gene_annotation()
            assert mock.call_count == 0  # Not called!
            assert len(annot) == 2

    def test_force_download_refreshes_cache(self, tmp_path, monkeypatch):
        """Test that force_download=True bypasses cache."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        with patch("deepscenic._datasets._fetch_from_biomart") as mock:
            mock_annot = pd.DataFrame(
                {"col": [1]},
                index=pd.Index(["A"], name="Gene"),
            )
            mock.return_value = (mock_annot, None)

            # First call
            ds.fetch_gene_annotation()

            # Second call with force_download
            ds.fetch_gene_annotation(force_download=True)

            assert mock.call_count == 2  # Called both times

    def test_different_params_create_different_cache(self, tmp_path, monkeypatch):
        """Test that different parameters create separate cache files."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        with patch("deepscenic._datasets._fetch_from_biomart") as mock:
            mock_annot = pd.DataFrame(
                {"col": [1]},
                index=pd.Index(["A"], name="Gene"),
            )
            mock.return_value = (mock_annot, None)

            # Call with different species
            ds.fetch_gene_annotation(species="hsapiens")
            ds.fetch_gene_annotation(species="mmusculus")

        # Should have 2 different cache files
        cache_files = list(tmp_path.glob("gene_annot_*.parquet"))
        assert len(cache_files) == 2


class TestCacheManagement:
    """Tests for cache management utilities."""

    def test_get_cache_info(self, tmp_path, monkeypatch):
        """Test cache info returns correct structure."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        # Create some fake cache files
        (tmp_path / "gene_annot_test.parquet").write_bytes(b"x" * 1000)
        (tmp_path / "allTFs_mouse.txt").write_text("TF1\nTF2\n")

        info = ds.get_cache_info()

        assert info["cache_dir"] == str(tmp_path)
        assert len(info["files"]) == 2
        assert info["total_size_mb"] > 0

    def test_get_cache_info_empty(self, tmp_path, monkeypatch):
        """Test cache info with empty cache."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        info = ds.get_cache_info()

        assert info["cache_dir"] == str(tmp_path)
        assert len(info["files"]) == 0
        assert info["total_size_mb"] == 0

    def test_clear_cache_all(self, tmp_path, monkeypatch):
        """Test clearing all cache files."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        # Create cache files
        (tmp_path / "file1.parquet").write_bytes(b"x")
        (tmp_path / "file2.txt").write_text("y")

        deleted = ds.clear_cache()

        assert len(deleted) == 2
        assert not (tmp_path / "file1.parquet").exists()
        assert not (tmp_path / "file2.txt").exists()

    def test_clear_cache_pattern(self, tmp_path, monkeypatch):
        """Test clearing cache with pattern."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)

        # Create cache files
        (tmp_path / "gene_annot_test.parquet").write_bytes(b"x")
        (tmp_path / "allTFs_mouse.txt").write_text("y")

        deleted = ds.clear_cache("gene_annot_*")

        assert len(deleted) == 1
        assert not (tmp_path / "gene_annot_test.parquet").exists()
        assert (tmp_path / "allTFs_mouse.txt").exists()  # Not deleted

    def test_clear_cache_nonexistent(self, tmp_path, monkeypatch):
        """Test clearing cache when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr(pooch, "os_cache", lambda _: nonexistent)

        deleted = ds.clear_cache()

        assert deleted == []


class TestFetchMMLines:
    """Tests for fetch_mm_lines function."""

    @staticmethod
    def _write_geo_files(tmp_path, rna_cells, atac_cells, meta_cells):
        """Write small GEO-style files: RNA as cells x genes, ATAC as regions x cells."""
        meta = pd.DataFrame(
            {"MMline": ["MM001", "MM001", "MM099"], "lineState": ["MEL", "MEL", "MES"]}, index=meta_cells
        )
        rna = pd.DataFrame(
            [[0.2 * i, 1.0, 0.0] for i in range(len(rna_cells))], index=rna_cells, columns=["SOX10", "MITF", "JUN"]
        )
        regions = ["chr1:100-600", "chr2:200-700"]
        atac = pd.DataFrame(
            [[float(int(c[1:])) + r for c in atac_cells] for r in range(len(regions))],
            index=regions,
            columns=atac_cells,
        )
        files = {}
        for key, df in [("metadata", meta), ("rna", rna), ("atac", atac)]:
            fname = MM_LINES_FILES[key][0]
            df.to_csv(tmp_path / fname, sep="\t", compression="gzip")
            files[fname] = str(tmp_path / fname)
        return files

    def test_files_have_checksums(self):
        """Test that every MM_lines file has a SHA256 checksum."""
        for fname, known_hash in MM_LINES_FILES.values():
            assert fname.startswith("GSE210745_mm_lines_")
            assert len(known_hash) == 64

    @patch("pooch.retrieve")
    def test_fetch_returns_aligned_data(self, mock_retrieve, tmp_path, monkeypatch):
        """Test shapes, orientation and cell order of the returned objects."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)
        files = self._write_geo_files(
            tmp_path, rna_cells=["c1", "c0", "c2"], atac_cells=["c2", "c0", "c1"], meta_cells=["c0", "c1", "c2"]
        )
        mock_retrieve.side_effect = lambda **kwargs: files[kwargs["fname"]]

        adata_rna, adata_atac, metadata = ds.fetch_mm_lines()

        assert mock_retrieve.call_count == 3
        assert list(metadata.columns) == ["MMline", "lineState"]
        assert list(adata_rna.obs_names) == list(adata_atac.obs_names) == list(metadata.index) == ["c0", "c1", "c2"]
        assert adata_rna.shape == (3, 3)
        assert adata_atac.shape == (3, 2)
        assert list(adata_atac.var_names) == ["chr1:100-600", "chr2:200-700"]
        assert adata_atac.X.dtype == np.float32
        # values follow their cells after reordering
        np.testing.assert_allclose(adata_rna[:, "SOX10"].X.ravel(), [0.2, 0.0, 0.4], rtol=1e-6)
        np.testing.assert_allclose(adata_atac.X[:, 1], [1.0, 2.0, 3.0])

    @patch("pooch.retrieve")
    def test_mismatched_cells_raises(self, mock_retrieve, tmp_path, monkeypatch):
        """Test error when the files describe different cells."""
        monkeypatch.setattr(pooch, "os_cache", lambda _: tmp_path)
        files = self._write_geo_files(
            tmp_path, rna_cells=["c0", "c1", "c2"], atac_cells=["c0", "c1", "c3"], meta_cells=["c0", "c1", "c2"]
        )
        mock_retrieve.side_effect = lambda **kwargs: files[kwargs["fname"]]

        with pytest.raises(ValueError, match="Cell names differ"):
            ds.fetch_mm_lines()

    def test_read_tsv_matrix_chunked(self, tmp_path):
        """Test that chunked reading matches a full read, with and without transposing."""
        df = pd.DataFrame(
            np.arange(35, dtype=float).reshape(7, 5), index=[f"r{i}" for i in range(7)], columns=list("abcde")
        )
        path = tmp_path / "matrix.tsv.gz"
        df.to_csv(path, sep="\t", compression="gzip")

        X, rows, cols = _read_tsv_matrix(str(path), chunksize=3)
        np.testing.assert_array_equal(X, df.to_numpy())
        assert list(rows) == list(df.index) and list(cols) == list(df.columns)

        Xt, rows_t, cols_t = _read_tsv_matrix(str(path), transpose=True, chunksize=3)
        np.testing.assert_array_equal(Xt, df.to_numpy().T)
        assert list(rows_t) == list(df.columns) and list(cols_t) == list(df.index)
