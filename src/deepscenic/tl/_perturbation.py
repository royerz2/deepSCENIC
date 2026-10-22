"""Perturbation simulation for deepSCENIC."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

from ._model import _check_mdata_matches_model

if TYPE_CHECKING:
    import mudata as md
    import pandas as pd

    from ._model import DeepSCENICModel


def simulate_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_name: str | list[str],
    level: float | list[float] = 0.0,
    n_iter: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
    clip_percentile: float = 99.99,
    return_intermediate: bool = False,
    *,
    mode: Literal["legacy", "counterfactual"] = "legacy",
    tol: float | None = None,
    damping: float = 1.0,
    return_diagnostics: bool = False,
    clip_max: float | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
    | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
):
    """
    Simulate one or more TF knockdowns/overexpressions.

    Perturbs a TF's expression and iteratively propagates the effect through
    the GRN to predict gene expression changes.

    The iteration is a fixed-point computation on the TF coordinates: only TF
    expression feeds back into the encoder. With the defaults, exactly
    ``n_iter`` steps are run and no convergence check is made (the original
    behaviour). Set ``tol`` to iterate until the state stops changing, and
    ``return_diagnostics=True`` to see whether it did.

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA + ATAC data
    tf_name
        TF name or list of TF names to perturb (e.g., ``"SOX10"`` or
        ``["SOX10", "MITF"]``).
    level
        Perturbation level or one level per TF. A scalar is applied to every
        TF (0 = knockout, >0 = overexpression).
    n_iter
        Number of iterations; with ``tol`` set, the maximum number.
    batch_size
        Cells per batch for inference
    device
        Device for computation
    split
        If specified, only process cells from this split
    clip_percentile
        Percentile for clipping perturbed values (default: 99.99)
    return_intermediate
        If True, return perturbed expression and logFC at each iteration.
        Cannot be combined with ``tol`` or ``return_diagnostics``.
    mode
        How the latent change is decoded into an expression change.
        ``"legacy"`` decodes the latent difference, ``g(z_pert - z_orig)``.
        ``"counterfactual"`` keeps each gene's own operating point,
        ``g(z_pert) - g(z_orig)``, i.e. the cell's residual
        ``x - g(z_orig)`` is held fixed (abduction-action-prediction). The two
        agree only where the decoder is linear.
    tol
        If given, stop once every cell's relative change per iteration,
        ``||x_k - x_{k-1}|| / ||x_k - x_0||`` over the non-perturbed genes, is
        at most ``tol``. A warning is raised if some perturbed cells (``support`` > 0,
        see ``return_diagnostics``) have not converged after ``n_iter`` iterations;
        cells where the TF already sits at the perturbation level are not counted.
    damping
        Step size ``a`` in ``(0, 1]``: ``x_k = (1 - a) x_{k-1} + a F(x_{k-1})``.
        ``a = 1`` is the original update. Damping does not change the fixed
        points; it can make the iteration converge where the plain update
        overshoots (see :func:`perturbation_stability`).
    return_diagnostics
        If True, also return a dict of per-cell arrays: ``"converged"``,
        ``"n_iter"`` (iterations used), ``"rel_change"`` (last relative
        change) and ``"support"``, the change in latent TF activity that the
        perturbation imposes, ``sum_j |phi(x_j) - phi(level_j)|``. A support
        of 0 means the TF is already at the perturbation level in that cell,
        and the predicted change is then 0 by construction: the model has no
        information about that perturbation in that cell.
    clip_max
        Upper clip value. By default it is the ``clip_percentile`` of the
        cells being simulated, so it changes with the cells passed in; pass
        the training data's value (e.g. ``np.percentile(train_X, 99.99)``) to
        simulate new or few cells on the same scale.

    Returns
    -------
    tuple
        ``(perturbed_matrix, logFC)``, both ``(n_cells, n_genes)``; with
        ``return_diagnostics``, ``(perturbed_matrix, logFC, diagnostics)``; with
        ``return_intermediate``, two dictionaries mapping each 1-indexed
        iteration to its perturbed matrix and logFC array.

    Examples
    --------
    >>> model = ds.tl.load_model("model.pt")
    >>> perturbed, logFC = ds.tl.simulate_perturbation(
    ...     model, mdata, tf_name="SOX10", level=0
    ... )

    >>> # Iterate to a checked steady state, with the counterfactual decoder
    >>> perturbed, logFC, diag = ds.tl.simulate_perturbation(
    ...     model, mdata, tf_name="SOX10", n_iter=200, tol=1e-4,
    ...     mode="counterfactual", return_diagnostics=True,
    ... )
    >>> diag["converged"].mean(), (diag["support"] > 0).mean()
    """
    if mode not in ("legacy", "counterfactual"):
        raise ValueError("mode must be 'legacy' or 'counterfactual'")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    if tol is not None and tol <= 0:
        raise ValueError("tol must be positive")
    if return_intermediate and (tol is not None or return_diagnostics):
        raise ValueError("return_intermediate cannot be combined with tol or return_diagnostics")

    model.eval()

    if device is None:
        device = next(model.vae.parameters()).device
    else:
        model.to(device)

    tf_names = [tf_name] if isinstance(tf_name, str) else list(tf_name)
    if not tf_names:
        raise ValueError("tf_name must contain at least one TF")
    if len(set(tf_names)) != len(tf_names):
        raise ValueError("tf_name contains duplicate TF names")

    if isinstance(level, (int, float)):
        levels = [float(level)] * len(tf_names)
    else:
        levels = [float(value) for value in level]
        if len(levels) != len(tf_names):
            raise ValueError("level must have the same length as tf_name")

    _check_mdata_matches_model(model, mdata)

    tf_indices = []
    for name in tf_names:
        if name not in model.tf_names:
            raise ValueError(f"TF '{name}' not found. Available TFs: {model.tf_names[:10]}...")
        tf_indices.append(model.tf_names.index(name))
    tf_gene_indices = [model.vae.tf_indices[index] for index in tf_indices]  # type: ignore[index]

    # Get data
    if split is not None:
        mask = mdata.obs["split"] == split
        rna = mdata.mod["rna"][mask].X
    else:
        rna = mdata.mod["rna"].X

    n_cells = rna.shape[0]

    # Compute clipping threshold from original data (unless given). Sparse data is densified only for
    # this (a temporary copy that may be partitioned in place); batches are densified below.
    if clip_max is None:
        if issparse(rna):
            clip_max = np.percentile(rna.toarray(), clip_percentile, overwrite_input=True)
        else:
            clip_max = np.percentile(rna, clip_percentile)

    # Genes whose change is tracked for convergence (everything except the clamped TFs)
    free_genes = torch.ones(rna.shape[1], dtype=torch.bool, device=device)
    for tf_gene_idx in tf_gene_indices:
        free_genes[tf_gene_idx] = False
    diag_parts: dict[str, list[np.ndarray]] = {"converged": [], "n_iter": [], "rel_change": [], "support": []}
    n_not_converged = 0
    n_perturbed = 0

    # Output arrays, allocated once the first batch shows their width and dtype
    outputs: dict[object, np.ndarray] = {}

    def _store(key: object, rows: slice, values: np.ndarray) -> None:
        if key not in outputs:
            outputs[key] = np.empty((n_cells, values.shape[1]), dtype=values.dtype)
        outputs[key][rows] = values

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Simulating perturbation"):
            end_idx = min(i + batch_size, n_cells)
            rows = slice(i, end_idx)
            batch_rna = rna[i:end_idx]
            batch_rna = batch_rna.toarray() if issparse(batch_rna) else batch_rna.copy()

            x_rna = torch.FloatTensor(batch_rna).to(device)

            # Get original z_rna (baseline gene regulatory signal)
            # skip_atac=True: only z_rna is needed, skipping ATAC decoder
            # saves ~37 GB VRAM per MLP layer (281k regions × 128 hidden)
            output_orig = model.vae(
                x_rna,
                model.adj_tf2r,
                use_mean=True,
                skip_atac=True,
            )
            z_rna_orig = output_orig.z_rna
            decoded_orig = model.vae.decoder_rna(z_rna_orig) if mode == "counterfactual" else None

            # Create perturbed expression matrix
            perturbed_rna = x_rna.clone()

            # Initialize logFC_decoded for edge case of n_iter=0
            logFC_decoded = torch.zeros_like(x_rna)

            for tf_gene_idx, perturbation_level in zip(tf_gene_indices, levels, strict=True):
                perturbed_rna[:, tf_gene_idx] = perturbation_level

            support = None
            if return_diagnostics or tol is not None:
                # Support: change in latent TF activity imposed by the clamp (the encoder acts per TF)
                x_tfs = x_rna[:, model.vae.tf_indices]
                x_tfs_level = x_tfs.clone()
                for tf_index, perturbation_level in zip(tf_indices, levels, strict=True):
                    x_tfs_level[:, tf_index] = perturbation_level
                phi = model.vae.encoder(x_tfs, use_mean=True)[0]
                phi_level = model.vae.encoder(x_tfs_level, use_mean=True)[0]
                support = (phi[:, tf_indices] - phi_level[:, tf_indices]).abs().sum(dim=1)
                if return_diagnostics:
                    diag_parts["support"].append(support.cpu().numpy())

            n_batch = end_idx - i
            converged = torch.zeros(n_batch, dtype=torch.bool, device=device)
            iters_used = torch.full((n_batch,), n_iter, dtype=torch.long, device=device)
            rel_change = torch.full((n_batch,), float("nan"), device=device)

            # Iterate towards a steady state
            for iter_num in range(1, n_iter + 1):
                # Set TF expressions to their perturbation levels
                for tf_gene_idx, perturbation_level in zip(tf_gene_indices, levels, strict=True):
                    perturbed_rna[:, tf_gene_idx] = perturbation_level

                # Forward pass with perturbed input (skip ATAC decoder)
                output_pert = model.vae(
                    perturbed_rna,
                    model.adj_tf2r,
                    use_mean=True,
                    skip_atac=True,
                )

                if mode == "legacy":
                    # Compute logFC in latent space, then decode the DIFFERENCE
                    # This matches legacy: decode(z_pert - z_orig), NOT decode(z_pert) - decode(z_orig)
                    logFC_latent = output_pert.z_rna - z_rna_orig
                    logFC_decoded = model.vae.decoder_rna(logFC_latent)
                else:
                    # Counterfactual: keep each gene's residual, move along the decoder from its own operating point
                    logFC_decoded = model.vae.decoder_rna(output_pert.z_rna) - decoded_orig

                # Update perturbed matrix: original + decoded fold change
                previous_rna = perturbed_rna
                perturbed_rna = x_rna + logFC_decoded

                # Clip to valid range [0, p99]
                perturbed_rna = torch.clamp(perturbed_rna, 0, clip_max)

                if damping < 1.0:
                    perturbed_rna = (1.0 - damping) * previous_rna + damping * perturbed_rna

                # Re-enforce TF perturbation levels after clipping
                for tf_gene_idx, perturbation_level in zip(tf_gene_indices, levels, strict=True):
                    perturbed_rna[:, tf_gene_idx] = perturbation_level

                if tol is not None or return_diagnostics:
                    step = (perturbed_rna - previous_rna)[:, free_genes].norm(dim=1)
                    size = (perturbed_rna - x_rna)[:, free_genes].norm(dim=1).clamp_min(1e-12)
                    rel_change = step / size
                    if tol is not None:
                        # Converged means "the last step was within tol", re-evaluated at every iteration: a cell
                        # that dipped below tol earlier but is moving again is not converged.
                        now = rel_change <= tol
                        iters_used = torch.where(now & ~converged, torch.full_like(iters_used, iter_num), iters_used)
                        iters_used = torch.where(~now, torch.full_like(iters_used, n_iter), iters_used)
                        converged = now
                        if bool(converged.all()):
                            break

                if return_intermediate:
                    _store(("perturbed", iter_num), rows, perturbed_rna.cpu().numpy())
                    _store(("logFC", iter_num), rows, logFC_decoded.cpu().numpy())

            if not return_intermediate:
                _store("perturbed", rows, perturbed_rna.cpu().numpy())
                _store("logFC", rows, logFC_decoded.cpu().numpy())
            if tol is not None:
                # Only cells that are actually perturbed (support > 0) count: where the TF already sits at the
                # clamp level there is nothing to converge, and the relative change is rounding noise over noise.
                n_not_converged += int((~converged & (support > 1e-6)).sum())
                n_perturbed += int((support > 1e-6).sum())
            if return_diagnostics:
                if tol is None:
                    converged = torch.zeros(n_batch, dtype=torch.bool, device=device)
                diag_parts["converged"].append(converged.cpu().numpy())
                diag_parts["n_iter"].append(iters_used.cpu().numpy())
                diag_parts["rel_change"].append(rel_change.cpu().numpy())

    if return_intermediate:
        perturbed_trace = {i: outputs[("perturbed", i)] for i in range(1, n_iter + 1)}
        logFC_trace = {i: outputs[("logFC", i)] for i in range(1, n_iter + 1)}
        return perturbed_trace, logFC_trace
    if n_not_converged:
        warnings.warn(
            f"{n_not_converged} of {n_perturbed} perturbed cells did not converge within n_iter={n_iter} (tol={tol}). "
            "Their result depends on the iteration count; see perturbation_stability().",
            RuntimeWarning,
            stacklevel=2,
        )
    if return_diagnostics:
        diagnostics = {key: np.concatenate(parts, axis=0) for key, parts in diag_parts.items()}
        return outputs["perturbed"], outputs["logFC"], diagnostics
    return outputs["perturbed"], outputs["logFC"]


def simulate_multi_perturbation(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_names: list[str],
    levels: list[float] | None = None,
    n_iter: int = 10,
    batch_size: int = 256,
    device: str | torch.device | None = None,
    split: str | None = None,
    clip_percentile: float = 99.99,
    return_intermediate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Simulate multiple TF perturbations.

    This compatibility wrapper delegates to ``simulate_perturbation``.
    New code can pass a list directly through ``tf_name`` instead.
    """
    perturbation_levels = [0.0] * len(tf_names) if levels is None else levels
    return simulate_perturbation(
        model=model,
        mdata=mdata,
        tf_name=tf_names,
        level=perturbation_levels,
        n_iter=n_iter,
        batch_size=batch_size,
        device=device,
        split=split,
        clip_percentile=clip_percentile,
        return_intermediate=return_intermediate,
    )


def _tf_update(
    model: DeepSCENICModel,
    x0: torch.Tensor,
    z0: torch.Tensor,
    g0: torch.Tensor | None,
    y: torch.Tensor,
    tf_cols: torch.Tensor,
    mode: str,
    clip_max: float,
    clamped: list[int],
    levels: list[float],
) -> torch.Tensor:
    """One application of the perturbation update, restricted to the TF coordinates ``y``.

    Only TF expression feeds back into the encoder, so iterating this map on ``y`` (with the other genes held at
    their measured values ``x0``) is the same iteration as :func:`simulate_perturbation`, restricted to TFs.
    """
    x = x0.clone()
    x[:, tf_cols] = y
    z = model.vae(x, model.adj_tf2r, use_mean=True, skip_atac=True).z_rna[:, tf_cols]
    dec = model.vae.decoder_rna(z - z0) if mode == "legacy" else model.vae.decoder_rna(z) - g0
    new = torch.clamp(x0[:, tf_cols] + dec, 0, clip_max)
    for t, v in zip(clamped, levels, strict=True):
        new[:, t] = v
    return new


def perturbation_stability(
    model: DeepSCENICModel,
    mdata: md.MuData,
    tf_name: str | list[str] | None = None,
    level: float | list[float] = 0.0,
    *,
    cells: np.ndarray | list[int] | None = None,
    mode: Literal["legacy", "counterfactual"] = "legacy",
    n_power: int = 60,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    clip_percentile: float = 99.99,
    seed: int = 0,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    Estimate the loop gain of the perturbation iteration for each cell.

    :func:`simulate_perturbation` iterates a map ``F`` on the TF coordinates.
    Near a state ``x``, errors evolve as ``d_{k+1} = J d_k`` with ``J`` the
    Jacobian of ``F`` on the free TF coordinates. Its spectral radius
    ``rho(J)`` decides what the iteration does: for ``rho < 1`` it settles
    (and forgets small errors); for ``rho > 1`` small differences grow, so the
    result depends on ``n_iter`` and the plain iteration does not reach a
    steady state (damping in :func:`simulate_perturbation` may help).

    ``rho`` is estimated at each cell's measured state, with the TFs in
    ``tf_name`` (if any) clamped to ``level`` and excluded from the free
    coordinates, by power iteration with exact Jacobian-vector products
    (forward-mode autograd): ``rho ~ exp(mean_k log ||J v_k||)`` over the second
    half of ``n_power`` normalised steps (Gelfand's formula). The estimate
    converges to ``rho`` as ``n_power`` grows; with a complex-conjugate leading
    pair of eigenvalues it converges more slowly. Where the update is clipped, the
    Jacobian is the one-sided autograd derivative (zero for values held at the
    clip boundary).

    Parameters
    ----------
    model
        Trained DeepSCENICModel
    mdata
        MuData with RNA data
    tf_name
        Optional TF(s) to clamp, as in :func:`simulate_perturbation`.
    level
        Clamp level(s) for ``tf_name``.
    cells
        Indices of the cells to evaluate (default: all).
    mode
        Decoder mode, as in :func:`simulate_perturbation`.
    n_power
        Power-iteration steps.
    batch_size
        Cells per batch.
    device
        Device for computation.
    clip_percentile
        Clipping percentile, as in :func:`simulate_perturbation`.
    seed
        Seed for the random start vectors.
    clip_max
        Upper clip value, as in :func:`simulate_perturbation`.

    Returns
    -------
    np.ndarray
        Estimated spectral radius per evaluated cell.

    Examples
    --------
    >>> rho = ds.tl.perturbation_stability(model, mdata, cells=range(50))
    >>> (rho < 1).mean()  # fraction of cells where the iteration contracts
    """
    model.eval()
    if device is None:
        device = next(model.vae.parameters()).device

    _check_mdata_matches_model(model, mdata)
    rna = mdata.mod["rna"].X
    if clip_max is None:
        clip_max = float(np.percentile(rna.toarray() if issparse(rna) else rna, clip_percentile))
    idx = np.arange(rna.shape[0]) if cells is None else np.asarray(cells)

    tf_cols = model.vae.tf_indices.to(device).long()
    clamped: list[int] = []
    levels: list[float] = []
    if tf_name is not None:
        names = [tf_name] if isinstance(tf_name, str) else list(tf_name)
        levels = [float(level)] * len(names) if isinstance(level, (int, float)) else [float(v) for v in level]
        if len(levels) != len(names):
            raise ValueError("level must have the same length as tf_name")
        for name in names:
            if name not in model.tf_names:
                raise ValueError(f"TF '{name}' not found.")
            clamped.append(model.tf_names.index(name))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = rna[idx[start : start + batch_size]]
            batch = batch.toarray() if issparse(batch) else np.asarray(batch)
            x0 = torch.tensor(batch, dtype=torch.float32, device=device)
            z0 = model.vae(x0, model.adj_tf2r, use_mean=True, skip_atac=True).z_rna[:, tf_cols]
            g0 = model.vae.decoder_rna(z0) if mode == "counterfactual" else None
            y0 = x0[:, tf_cols].clone()
            for t, v in zip(clamped, levels, strict=True):
                y0[:, t] = v

            def update(y: torch.Tensor, x0=x0, z0=z0, g0=g0) -> torch.Tensor:
                return _tf_update(model, x0, z0, g0, y, tf_cols, mode, clip_max, clamped, levels)

            vec = torch.randn(y0.shape, generator=generator).to(device)
            vec[:, clamped] = 0.0
            vec = vec / vec.norm(dim=1, keepdim=True)
            logs = []
            for _ in range(n_power):
                _, jv = torch.func.jvp(update, (y0,), (vec,))
                jv[:, clamped] = 0.0
                norm = jv.norm(dim=1, keepdim=True).clamp_min(1e-30)
                logs.append(torch.log(norm[:, 0]))
                vec = jv / norm
            out.append(torch.exp(torch.stack(logs[n_power // 2 :]).mean(dim=0)).cpu().numpy())
    return np.concatenate(out)


def process_perturbation_results(
    logFC: np.ndarray,
    mdata: md.MuData,
    tf_name: str | list[str],
    *,
    gene_subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Process perturbation results into a DataFrame for plotting.

    Computes mean log fold change across cells for each gene. The values
    represent the decoder-space effect of the perturbation (not literal
    log2 fold changes).

    Parameters
    ----------
    logFC
        Log fold change array from simulate_perturbation (n_cells, n_genes)
    mdata
        MuData used for simulation (provides gene names)
    tf_name
        Name of perturbed TF(s) - used for the 'tf' column
    gene_subset
        If provided, only include these genes in output

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'gene', 'tf', 'log2fc', 'abs_log2fc'.
        Ready for waterfall_perturbation, heatmap_perturbation, or
        dotplot_perturbation.

    Examples
    --------
    >>> perturbed, logFC = ds.tl.simulate_perturbation(model, mdata, "SOX10")
    >>> results = ds.tl.process_perturbation_results(logFC, mdata, "SOX10")
    >>> ds.pl.waterfall_perturbation(results)

    >>> # For multi-TF comparison heatmaps
    >>> results_list = []
    >>> for tf in ["SOX10", "MITF", "PAX3"]:
    ...     _, logFC = ds.tl.simulate_perturbation(model, mdata, tf)
    ...     results_list.append(ds.tl.process_perturbation_results(logFC, mdata, tf))
    >>> combined = pd.concat(results_list)
    >>> ds.pl.heatmap_perturbation(combined)
    """
    import pandas as pd

    # Get gene names from mdata
    gene_names = list(mdata.mod["rna"].var_names)

    if logFC.shape[1] != len(gene_names):
        raise ValueError(f"logFC has {logFC.shape[1]} genes but mdata has {len(gene_names)} genes")

    # Handle tf_name as string or list
    if isinstance(tf_name, str):
        tf_label = tf_name
    else:
        tf_label = "+".join(tf_name)  # e.g., "SOX10+MITF" for multi-TF

    # Compute mean logFC per gene (across cells)
    mean_logfc = logFC.mean(axis=0)

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "gene": gene_names,
            "tf": tf_label,
            "log2fc": mean_logfc,
            "abs_log2fc": np.abs(mean_logfc),
        }
    )

    # Filter to gene subset if provided
    if gene_subset is not None:
        result = result[result["gene"].isin(gene_subset)].copy()

    return result
