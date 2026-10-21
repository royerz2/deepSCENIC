"""Tests for perturbation functions (_perturbation.py)."""

import numpy as np
import pandas as pd
import pytest

from deepscenic.tl._perturbation import (
    process_perturbation_results,
    simulate_multi_perturbation,
    simulate_perturbation,
)


class TestSimulatePerturbation:
    """Tests for simulate_perturbation function."""

    def test_returns_numpy_array(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return numpy array."""
        perturbed, logFC = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        assert isinstance(perturbed, np.ndarray)
        assert isinstance(logFC, np.ndarray)

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output should have shape (n_cells, n_genes)."""
        perturbed, logFC = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, n_iter=2, batch_size=4, device="cpu"
        )
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        assert perturbed.shape == (n_cells, d["n_genes"])
        assert logFC.shape == (n_cells, d["n_genes"])

    def test_invalid_tf_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise error for invalid TF name."""
        with pytest.raises(ValueError, match="not found"):
            simulate_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_name="INVALID_TF",
                level=0.0,
                n_iter=2,
                batch_size=4,
                device="cpu",
            )

    def test_return_intermediate(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should return dict of iterations when return_intermediate=True."""
        n_iter = 3
        perturbed_trace, logFC_trace = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            level=0.0,
            n_iter=n_iter,
            batch_size=4,
            device="cpu",
            return_intermediate=True,
        )
        assert isinstance(perturbed_trace, dict)
        assert isinstance(logFC_trace, dict)
        assert set(perturbed_trace.keys()) == {1, 2, 3}
        assert set(logFC_trace.keys()) == {1, 2, 3}
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        for i in range(1, n_iter + 1):
            assert perturbed_trace[i].shape == (n_cells, d["n_genes"])
            assert logFC_trace[i].shape == (n_cells, d["n_genes"])

    def test_multiple_tfs(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should accept multiple TFs and one level per TF."""
        perturbed, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name=["TF0", "TF1"],
            level=[0.0, 2.0],
            n_iter=2,
            batch_size=4,
            device="cpu",
        )

        tf0_idx = int(mock_deepscenic_model.vae.tf_indices[0])
        tf1_idx = int(mock_deepscenic_model.vae.tf_indices[1])
        assert np.all(perturbed[:, tf0_idx] == 0.0)
        assert np.all(perturbed[:, tf1_idx] == 2.0)
        assert perturbed.shape == logFC.shape

    def test_mismatched_levels_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Test that function raises error when levels and TF names have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            simulate_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_name=["TF0", "TF1"],
                level=[0.0],
                device="cpu",
            )


class TestSimulateMultiPerturbation:
    """Tests for simulate_multi_perturbation function."""

    def test_returns_numpy_array(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return numpy array."""
        perturbed, logFC = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        assert isinstance(perturbed, np.ndarray)
        assert isinstance(logFC, np.ndarray)

    def test_output_shape(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Output should have shape (n_cells, n_genes)."""
        perturbed, logFC = simulate_multi_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, tf_names=["TF0", "TF1"], n_iter=2, batch_size=4, device="cpu"
        )
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        assert perturbed.shape == (n_cells, d["n_genes"])
        assert logFC.shape == (n_cells, d["n_genes"])

    def test_invalid_tf_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise error for invalid TF name."""
        with pytest.raises(ValueError, match="not found"):
            simulate_multi_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_names=["TF0", "INVALID"],
                n_iter=2,
                batch_size=4,
                device="cpu",
            )

    def test_custom_levels(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should accept custom levels per TF."""
        result = simulate_multi_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_names=["TF0", "TF1"],
            levels=[0.0, 2.0],
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        assert all(value is not None for value in result)

    def test_mismatched_levels_raises(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should raise error if levels length doesn't match tf_names."""
        with pytest.raises(ValueError, match="same length"):
            simulate_multi_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                tf_names=["TF0", "TF1"],
                levels=[0.0],
                n_iter=2,
                batch_size=4,
                device="cpu",
            )

    def test_return_intermediate(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should return dict of iterations when return_intermediate=True."""
        n_iter = 3
        perturbed_trace, logFC_trace = simulate_multi_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_names=["TF0", "TF1"],
            n_iter=n_iter,
            batch_size=4,
            device="cpu",
            return_intermediate=True,
        )
        assert isinstance(perturbed_trace, dict)
        assert isinstance(logFC_trace, dict)
        assert set(perturbed_trace.keys()) == {1, 2, 3}
        assert set(logFC_trace.keys()) == {1, 2, 3}
        d = minimal_dims
        n_cells = d["n_cells"] * 2
        for i in range(1, n_iter + 1):
            assert perturbed_trace[i].shape == (n_cells, d["n_genes"])
            assert logFC_trace[i].shape == (n_cells, d["n_genes"])


class TestProcessPerturbationResults:
    """Tests for process_perturbation_results function."""

    def test_returns_dataframe(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should return pandas DataFrame."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should have expected columns."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert "gene" in result.columns
        assert "tf" in result.columns
        assert "log2fc" in result.columns
        assert "abs_log2fc" in result.columns

    def test_output_rows(self, mock_deepscenic_model, mock_mdata_for_model, minimal_dims):
        """Should have one row per gene."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert len(result) == minimal_dims["n_genes"]

    def test_no_pvalue_columns(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should not include pvalue/padj columns (removed from API)."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0")

        assert "pvalue" not in result.columns
        assert "padj" not in result.columns

    def test_gene_subset(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should filter to gene subset."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )

        gene_names = list(mock_mdata_for_model.mod["rna"].var_names[:3])
        result = process_perturbation_results(logFC, mock_mdata_for_model, "TF0", gene_subset=gene_names)

        assert len(result) == 3
        assert set(result["gene"]) == set(gene_names)

    def test_multi_tf_label(self, mock_deepscenic_model, mock_mdata_for_model):
        """Should combine multiple TF names with '+'."""
        _, logFC = simulate_perturbation(
            mock_deepscenic_model,
            mock_mdata_for_model,
            tf_name="TF0",
            n_iter=2,
            batch_size=4,
            device="cpu",
        )
        result = process_perturbation_results(logFC, mock_mdata_for_model, ["TF0", "TF1"])

        assert result["tf"].iloc[0] == "TF0+TF1"

    def test_mismatched_shape_raises(self, mock_mdata_for_model):
        """Should raise error if logFC shape doesn't match mdata."""
        logFC = np.zeros((10, 5))  # Wrong number of genes

        with pytest.raises(ValueError, match="genes"):
            process_perturbation_results(logFC, mock_mdata_for_model, "TF0")


class TestCausalPerturbationOptions:
    """Opt-in convergence, counterfactual and diagnostics options (defaults unchanged)."""

    def _sim(self, model, mdata, **kw):
        args = {"tf_name": "TF0", "level": 0.0, "n_iter": 3, "batch_size": 4, "device": "cpu"}
        args.update(kw)
        return simulate_perturbation(model, mdata, **args)

    def test_defaults_unchanged_by_diagnostics(self, mock_deepscenic_model, mock_mdata_for_model):
        """Asking for diagnostics must not change the prediction."""
        p1, l1 = self._sim(mock_deepscenic_model, mock_mdata_for_model)
        p2, l2, diag = self._sim(mock_deepscenic_model, mock_mdata_for_model, return_diagnostics=True)
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        np.testing.assert_allclose(l1, l2, atol=1e-6)
        assert set(diag) == {"converged", "n_iter", "rel_change", "support"}
        assert all(v.shape == (p1.shape[0],) for v in diag.values())

    def test_damping_one_is_legacy(self, mock_deepscenic_model, mock_mdata_for_model):
        """damping=1 is the original update."""
        _, l1 = self._sim(mock_deepscenic_model, mock_mdata_for_model)
        _, l2 = self._sim(mock_deepscenic_model, mock_mdata_for_model, damping=1.0)
        np.testing.assert_allclose(l1, l2, atol=1e-6)

    def test_counterfactual_differs_from_legacy(self, mock_deepscenic_model, mock_mdata_for_model):
        """With a nonlinear decoder the counterfactual update differs from decoding the difference."""
        _, l_legacy = self._sim(mock_deepscenic_model, mock_mdata_for_model, n_iter=1)
        _, l_cf = self._sim(mock_deepscenic_model, mock_mdata_for_model, n_iter=1, mode="counterfactual")
        assert not np.allclose(l_legacy, l_cf)

    def test_zero_support_gives_zero_change(self, mock_deepscenic_model, mock_mdata_for_model):
        """A knockout of a TF that is already 0 changes nothing, in both modes, and has support 0."""
        mdata = mock_mdata_for_model.copy()
        tf_col = int(mock_deepscenic_model.vae.tf_indices[0])
        mdata.mod["rna"].X[:, tf_col] = 0.0
        for mode in ("legacy", "counterfactual"):
            _, logfc, diag = self._sim(mock_deepscenic_model, mdata, mode=mode, return_diagnostics=True)
            np.testing.assert_allclose(diag["support"], 0.0, atol=1e-7)
            np.testing.assert_allclose(logfc, 0.0, atol=1e-6)

    def test_tol_reports_convergence(self, mock_deepscenic_model, mock_mdata_for_model):
        """With tol, iterations used are reported and bounded by n_iter."""
        _, _, diag = self._sim(
            mock_deepscenic_model, mock_mdata_for_model, n_iter=200, tol=1e-3, damping=0.5, return_diagnostics=True
        )
        assert diag["n_iter"].max() <= 200
        assert np.all(diag["rel_change"][diag["converged"]] <= 1e-3)

    def test_warns_when_not_converged(self, mock_deepscenic_model, mock_mdata_for_model):
        """One iteration with a tiny tolerance cannot converge: warn."""
        with pytest.warns(RuntimeWarning, match="did not converge"):
            self._sim(mock_deepscenic_model, mock_mdata_for_model, n_iter=1, tol=1e-12)

    def test_invalid_options_raise(self, mock_deepscenic_model, mock_mdata_for_model):
        """Bad option values and incompatible combinations are rejected."""
        with pytest.raises(ValueError, match="mode"):
            self._sim(mock_deepscenic_model, mock_mdata_for_model, mode="other")
        with pytest.raises(ValueError, match="damping"):
            self._sim(mock_deepscenic_model, mock_mdata_for_model, damping=0.0)
        with pytest.raises(ValueError, match="return_intermediate"):
            self._sim(mock_deepscenic_model, mock_mdata_for_model, tol=1e-3, return_intermediate=True)


class TestPerturbationStability:
    """Tests for perturbation_stability (loop-gain estimate)."""

    def test_shape_and_values(self, mock_deepscenic_model, mock_mdata_for_model):
        """One finite, non-negative spectral radius per evaluated cell."""
        from deepscenic.tl._perturbation import perturbation_stability

        rho = perturbation_stability(mock_deepscenic_model, mock_mdata_for_model, cells=[0, 1, 2], device="cpu")
        assert rho.shape == (3,)
        assert np.all(np.isfinite(rho)) and np.all(rho >= 0)

    def test_matches_exact_jacobian_on_linear_map(self):
        """On a known linear map the power-iteration estimate recovers the spectral radius."""
        rng = np.random.default_rng(0)
        J = rng.normal(size=(6, 6)) * 0.4
        true_rho = np.abs(np.linalg.eigvals(J)).max()
        v = rng.normal(size=6)
        v /= np.linalg.norm(v)
        logs = []
        for _ in range(200):
            jv = J @ v
            logs.append(np.log(np.linalg.norm(jv)))
            v = jv / np.linalg.norm(jv)
        est = np.exp(np.mean(logs[100:]))
        assert abs(est - true_rho) / true_rho < 0.05

    def test_clamped_tf_with_level(self, mock_deepscenic_model, mock_mdata_for_model):
        """Clamping a TF is accepted; an unknown TF raises."""
        from deepscenic.tl._perturbation import perturbation_stability

        rho = perturbation_stability(
            mock_deepscenic_model, mock_mdata_for_model, tf_name="TF0", level=0.0, cells=[0, 1], device="cpu"
        )
        assert rho.shape == (2,)
        with pytest.raises(ValueError, match="not found"):
            perturbation_stability(mock_deepscenic_model, mock_mdata_for_model, tf_name="NOPE", device="cpu")


def test_clip_max_overrides_percentile(mock_deepscenic_model, mock_mdata_for_model):
    """An explicit clip_max equal to the percentile gives the same result; a lower one clips harder."""
    X = mock_mdata_for_model.mod["rna"].X
    c = float(np.percentile(X, 99.99))
    _, a = simulate_perturbation(mock_deepscenic_model, mock_mdata_for_model, "TF0", 2.0, n_iter=3, device="cpu")
    _, b = simulate_perturbation(
        mock_deepscenic_model, mock_mdata_for_model, "TF0", 2.0, n_iter=3, device="cpu", clip_max=c
    )
    p, _ = simulate_perturbation(
        mock_deepscenic_model, mock_mdata_for_model, "TF0", 2.0, n_iter=3, device="cpu", clip_max=0.1
    )
    np.testing.assert_allclose(a, b, atol=1e-6)
    tf_cols = mock_deepscenic_model.vae.tf_indices
    others = np.setdiff1d(np.arange(p.shape[1]), [int(tf_cols[0])])
    assert p[:, others].max() <= 0.1 + 1e-6
