"""Tests for perturbation functions (_perturbation.py)."""

import numpy as np
import pandas as pd
import pytest
import torch

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
        """With a nonlinear decoder the counterfactual update differs from decoding the difference.

        The mock model's weights are scaled so that the decoder works in its nonlinear range.
        """
        mock_deepscenic_model.adj_tf2r = mock_deepscenic_model.adj_tf2r * 300.0
        _, l_legacy = self._sim(mock_deepscenic_model, mock_mdata_for_model, n_iter=1)
        _, l_cf = self._sim(mock_deepscenic_model, mock_mdata_for_model, n_iter=1, mode="counterfactual")
        assert np.abs(l_legacy - l_cf).max() > 1e-2 * np.abs(l_legacy).max()

    def test_zero_support_gives_zero_change(self, mock_deepscenic_model, mock_mdata_for_model):
        """A knockout of a TF that is already 0 changes nothing, in both modes, and has support 0."""
        mdata = mock_mdata_for_model.copy()
        tf_col = int(mock_deepscenic_model.vae.tf_indices[0])
        mdata.mod["rna"].X[:, tf_col] = 0.0
        for mode in ("legacy", "counterfactual"):
            # clip_max is pinned: on a tiny mock dataset the default clip (a percentile of the data) can sit
            # below the largest value, and clipping that value would change the result slightly
            _, logfc, diag = self._sim(mock_deepscenic_model, mdata, mode=mode, return_diagnostics=True, clip_max=1e3)
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


def _reference_legacy(model, mdata, tf, level, n_iter, clip_max):
    """Independent re-implementation of the original (default) knockdown iteration, for comparison."""
    x0 = torch.tensor(np.asarray(mdata.mod["rna"].X), dtype=torch.float32)
    j = int(model.vae.tf_indices[model.tf_names.index(tf)])
    with torch.no_grad():
        z0 = model.vae(x0, model.adj_tf2r, use_mean=True, skip_atac=True).z_rna
        p = x0.clone()
        p[:, j] = level
        for _ in range(n_iter):
            p[:, j] = level
            z = model.vae(p, model.adj_tf2r, use_mean=True, skip_atac=True).z_rna
            dec = model.vae.decoder_rna(z - z0)
            p = torch.clamp(x0 + dec, 0, clip_max)
            p[:, j] = level
    return p.numpy(), dec.numpy()


class TestVersusReferenceAndExactQuantities:
    """The new options are checked against independent computations."""

    def test_default_matches_independent_reference(self, mock_deepscenic_model, mock_mdata_for_model):
        """Default output equals a from-scratch implementation of the original iteration."""
        c = float(np.percentile(mock_mdata_for_model.mod["rna"].X, 99.99))
        p_ref, l_ref = _reference_legacy(mock_deepscenic_model, mock_mdata_for_model, "TF0", 0.0, 4, c)
        p, lf = simulate_perturbation(
            mock_deepscenic_model, mock_mdata_for_model, "TF0", 0.0, n_iter=4, batch_size=5, device="cpu"
        )
        np.testing.assert_allclose(p, p_ref, atol=1e-6)
        np.testing.assert_allclose(lf, l_ref, atol=1e-6)

    def test_counterfactual_equals_legacy_for_linear_decoder(self, mock_deepscenic_model, mock_mdata_for_model):
        """If the decoder is linear, g(z_pert) - g(z_orig) = g(z_pert - z_orig): the two modes agree."""

        class Linear(torch.nn.Module):
            def forward(self, z):
                return 0.7 * z

        mock_deepscenic_model.vae.decoder_rna = Linear()
        kw = {"tf_name": "TF0", "level": 0.0, "n_iter": 5, "batch_size": 4, "device": "cpu"}
        _, a = simulate_perturbation(mock_deepscenic_model, mock_mdata_for_model, mode="legacy", **kw)
        _, b = simulate_perturbation(mock_deepscenic_model, mock_mdata_for_model, mode="counterfactual", **kw)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_support_positive_when_tf_expressed(self, mock_deepscenic_model, mock_mdata_for_model):
        """A TF expressed away from the knockdown level has support > 0 and a non-zero predicted change."""
        mdata = mock_mdata_for_model.copy()
        col = int(mock_deepscenic_model.vae.tf_indices[0])
        mdata.mod["rna"].X[:, col] = 2.0
        _, logfc, diag = simulate_perturbation(
            mock_deepscenic_model, mdata, "TF0", 0.0, n_iter=3, batch_size=4, device="cpu", return_diagnostics=True
        )
        assert np.all(diag["support"] > 0)
        assert np.abs(logfc).max() > 0

    def test_no_warning_when_converged(self, mock_deepscenic_model, mock_mdata_for_model):
        """A huge tolerance is met at once: no warning, every cell flagged converged."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _, _, diag = simulate_perturbation(
                mock_deepscenic_model,
                mock_mdata_for_model,
                "TF0",
                0.0,
                n_iter=20,
                tol=1e9,
                batch_size=4,
                device="cpu",
                return_diagnostics=True,
            )
        assert diag["converged"].all()

    def test_one_update_matches_one_simulated_step(self, mock_deepscenic_model, mock_mdata_for_model):
        """The TF-coordinate update used by perturbation_stability equals one step of simulate_perturbation."""
        from deepscenic.tl._perturbation import _tf_update

        m, d = mock_deepscenic_model, mock_mdata_for_model
        c = float(np.percentile(d.mod["rna"].X, 99.99))
        x0 = torch.tensor(np.asarray(d.mod["rna"].X), dtype=torch.float32)
        tf_cols = m.vae.tf_indices.long()
        for mode in ("legacy", "counterfactual"):
            with torch.no_grad():
                z0 = m.vae(x0, m.adj_tf2r, use_mean=True, skip_atac=True).z_rna[:, tf_cols]
                g0 = m.vae.decoder_rna(z0) if mode == "counterfactual" else None
                y0 = x0[:, tf_cols].clone()
                y0[:, 0] = 0.0
                y1 = _tf_update(m, x0, z0, g0, y0, tf_cols, mode, c, [0], [0.0])
            p, _ = simulate_perturbation(m, d, "TF0", 0.0, n_iter=1, batch_size=4, device="cpu", mode=mode, clip_max=c)
            np.testing.assert_allclose(y1.numpy(), p[:, tf_cols.numpy()], atol=1e-6)

    def test_stability_matches_exact_jacobian(self, mock_deepscenic_model, mock_mdata_for_model):
        """perturbation_stability agrees with the exact spectral radius (autograd Jacobian) within 5 %.

        The mock model's loop gain is tiny, so its weights are scaled until the gains are of order 1.
        """
        from deepscenic.tl._perturbation import _tf_update, perturbation_stability

        m, d = mock_deepscenic_model, mock_mdata_for_model
        m.adj_tf2r = m.adj_tf2r * 300.0
        c = float(np.percentile(d.mod["rna"].X, 99.99))
        tf_cols = m.vae.tf_indices.long()
        for mode in ("legacy", "counterfactual"):
            est = perturbation_stability(
                m, d, tf_name="TF0", level=0.0, cells=list(range(8)), mode=mode, n_power=200, device="cpu", clip_max=c
            )
            for i in range(8):
                x0 = torch.tensor(np.asarray(d.mod["rna"].X[[i]]), dtype=torch.float32)
                with torch.no_grad():
                    z0 = m.vae(x0, m.adj_tf2r, use_mean=True, skip_atac=True).z_rna[:, tf_cols]
                    g0 = m.vae.decoder_rna(z0) if mode == "counterfactual" else None
                y0 = x0[:, tf_cols].clone()
                y0[:, 0] = 0.0
                free = list(range(1, y0.shape[1]))

                def f(yf, x0=x0, z0=z0, g0=g0, y0=y0, free=free, mode=mode):
                    y = y0.clone()
                    y[0, free] = yf
                    return _tf_update(m, x0, z0, g0, y, tf_cols, mode, c, [0], [0.0])[0, free]

                J = torch.func.jacrev(f)(y0[0, free].clone())
                exact = float(torch.linalg.eigvals(J.detach().double()).abs().max())
                if exact > 0.05:  # below that (e.g. every value clipped) there is nothing meaningful to compare
                    assert abs(est[i] - exact) / exact < 0.05, (mode, i, est[i], exact)


def test_warning_ignores_unperturbed_cells(mock_deepscenic_model, mock_mdata_for_model):
    """Cells whose TF already sits at the knockdown level have nothing to converge: no warning for them."""
    import warnings

    mdata = mock_mdata_for_model.copy()
    mdata.mod["rna"].X[:, int(mock_deepscenic_model.vae.tf_indices[0])] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, _, diag = simulate_perturbation(
            mock_deepscenic_model,
            mdata,
            "TF0",
            0.0,
            n_iter=1,
            tol=1e-12,
            batch_size=4,
            device="cpu",
            return_diagnostics=True,
            clip_max=1e3,
        )
    assert np.all(diag["support"] == 0)
