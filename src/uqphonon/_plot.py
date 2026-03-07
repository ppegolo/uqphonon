"""Phonon band structure plotting for ensembles.

Provides four plot modes:
- "mean"      : mean band structure only
- "mean+std"  : mean ± std as a shaded region
- "ensemble"  : all ensemble members (light) + mean (bold)
- "spectral"  : 2D heatmap (spectral-function style)
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

# Conversion factors from phonopy's native THz
_UNIT_FACTORS = {
    "THz": 1.0,
    "cm-1": 33.35641,
    "meV": 4.13567,
}
_UNIT_LABELS = {
    "THz": "Frequency (THz)",
    "cm-1": "Frequency (cm⁻¹)",
    "meV": "Frequency (meV)",
}


def _get_unit_factor(unit: str) -> float:
    if unit not in _UNIT_FACTORS:
        raise ValueError(f"Unknown unit '{unit}'. Choose from {list(_UNIT_FACTORS)}.")
    return _UNIT_FACTORS[unit]


def _draw_band(ax, phonon, scale: float, **line_kwargs):
    """Draw all segments of one phonopy band structure on ax."""
    bs = phonon.band_structure
    for dist, freq, conn in zip(bs.distances, bs.frequencies, bs.path_connections):
        ax.plot(dist, freq * scale, **line_kwargs)


def _decorate_axes(ax, phonon, scale: float, unit: str):
    """Add high-symmetry point ticks, zero line, and axis labels."""
    bs = phonon.band_structure
    # Collect tick positions and labels
    tick_positions = []
    tick_labels = []
    distances = bs.distances
    path_connections = bs.path_connections
    labels = bs.labels

    label_idx = 0
    for i, (dist, conn) in enumerate(zip(distances, path_connections)):
        if i == 0:
            tick_positions.append(dist[0])
            tick_labels.append(labels[label_idx] if labels else "")
            label_idx += 1
        tick_positions.append(dist[-1])
        tick_labels.append(labels[label_idx] if labels else "")
        label_idx += 1
        if not conn and i < len(distances) - 1:
            # Segment break: next segment starts fresh, skip duplicating
            pass

    # Merge labels at coincident tick positions (segment breaks → "X|Y")
    merged_positions = []
    merged_labels = []
    tol = 1e-8
    for pos, lbl in zip(tick_positions, tick_labels):
        if merged_positions and abs(pos - merged_positions[-1]) < tol:
            if lbl and lbl != merged_labels[-1]:
                merged_labels[-1] = merged_labels[-1] + "|" + lbl
        else:
            merged_positions.append(pos)
            merged_labels.append(lbl)

    ax.set_xticks(merged_positions)
    ax.set_xticklabels(merged_labels)
    for pos in merged_positions[1:-1]:
        ax.axvline(pos, color="k", lw=0.5, ls=":")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlim(distances[0][0], distances[-1][-1])
    ax.set_ylabel(_UNIT_LABELS.get(unit, f"Frequency ({unit})"))
    ax.set_xlabel("Wavevector")


def _stack_frequencies(phonons) -> np.ndarray:
    """Stack frequencies from all ensemble phonons.

    Returns
    -------
    freqs : ndarray, shape (n_ensemble, n_segments, n_qpts, n_bands)
        Note: segments may have different npoints, so this returns a list of arrays.
    """
    return [
        np.stack([ph.band_structure.frequencies[i] for ph in phonons], axis=0)
        for i in range(len(phonons[0].band_structure.frequencies))
    ]


def _plot_spectral(ax, phonons, scale: float, cmap: str):
    """Draw a spectral-function-style heatmap with Gaussian broadening.

    Each band mode contributes a Gaussian peak of width sigma to the spectral
    weight S(q, ω). S is normalised per q-point (max = 1) and plotted with a
    shared colour scale across all segments.
    """
    bs = phonons[0].band_structure
    nfreq_pts = 300

    # Global frequency range
    all_freqs = (
        np.concatenate(
            [
                ph.band_structure.frequencies[i].ravel()
                for ph in phonons
                for i in range(len(bs.distances))
            ]
        )
        * scale
    )
    freq_min, freq_max = all_freqs.min(), all_freqs.max()
    freq_grid = np.linspace(freq_min, freq_max, nfreq_pts)  # (nfreq_pts,)
    sigma = (freq_max - freq_min) / 50  # ~2 % of the total range

    # Pre-compute S for all segments so we can set a shared vmax
    S_list = []
    d_edges_list = []

    for seg_idx, dist in enumerate(bs.distances):
        seg_freqs = (
            np.stack([ph.band_structure.frequencies[seg_idx] for ph in phonons], axis=0)
            * scale
        )  # (n_ensemble, n_qpts, n_bands)
        n_qpts = seg_freqs.shape[1]

        # freqs_at_q: (n_qpts, n_ensemble * n_bands)
        freqs_at_q = seg_freqs.transpose(1, 0, 2).reshape(n_qpts, -1)

        # S[iq, iω] = Σ_mode exp(-½ ((ω - ω_mode) / σ)²)
        # diff: (n_qpts, n_modes, nfreq_pts)
        diff = freq_grid[np.newaxis, np.newaxis, :] - freqs_at_q[:, :, np.newaxis]
        S = np.exp(-0.5 * (diff / sigma) ** 2).sum(axis=1)  # (n_qpts, nfreq_pts)

        # Normalise per q-point so every q has the same peak height
        q_max = S.max(axis=1, keepdims=True)
        q_max[q_max == 0] = 1
        S /= q_max

        if n_qpts > 1:
            d_edges = np.concatenate(
                [
                    [dist[0] - (dist[1] - dist[0]) / 2],
                    (dist[:-1] + dist[1:]) / 2,
                    [dist[-1] + (dist[-1] - dist[-2]) / 2],
                ]
            )
        else:
            d_edges = np.array([dist[0] - 0.01, dist[0] + 0.01])

        S_list.append(S)
        d_edges_list.append(d_edges)

    df = freq_grid[1] - freq_grid[0]
    freq_edges = np.concatenate(
        [
            [freq_grid[0] - df / 2],
            (freq_grid[:-1] + freq_grid[1:]) / 2,
            [freq_grid[-1] + df / 2],
        ]
    )

    for S, d_edges in zip(S_list, d_edges_list):
        ax.pcolormesh(
            d_edges, freq_edges, S.T, cmap=cmap, shading="auto", vmin=0, vmax=1
        )


def plot_bands(
    phonons,
    mean_phonon,
    mode: str = "ensemble",
    unit: str = "cm-1",
    ax: matplotlib.axes.Axes | None = None,
    color: str = "tab:red",
    ensemble_alpha: float = 0.05,
    ensemble_linewidth: float = 0.5,
    mean_linewidth: float = 1.5,
    std_alpha: float = 0.3,
    cmap: str = "inferno",
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot phonon band structure with UQ.

    Parameters
    ----------
    phonons : list of Phonopy
        One Phonopy object per ensemble member (with band structure computed).
    mean_phonon : Phonopy
        Phonopy object for the mean-force bands.
    mode : str
        One of "mean", "mean+std", "ensemble", or "spectral".
    unit : str
        Frequency unit: "THz", "cm-1", or "meV".
    ax : matplotlib Axes, optional
        If None, a new figure and axes are created.
    color : str
        Color for the MLIP bands (not used in "spectral" mode).
    ensemble_alpha : float
        Alpha for individual ensemble members (mode="ensemble").
    ensemble_linewidth : float
        Line width for ensemble members.
    mean_linewidth : float
        Line width for the mean band.
    std_alpha : float
        Alpha for the std fill (mode="mean+std").
    cmap : str
        Colormap for mode="spectral".

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if mode not in ("mean", "mean+std", "ensemble", "spectral"):
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'mean', 'mean+std', 'ensemble', or 'spectral'."
        )

    scale = _get_unit_factor(unit)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if mode == "ensemble":
        for ph in phonons:
            _draw_band(
                ax, ph, scale, color=color, lw=ensemble_linewidth, alpha=ensemble_alpha
            )
        _draw_band(ax, mean_phonon, scale, color=color, lw=mean_linewidth, alpha=1.0)

    elif mode == "mean+std":
        # Stack per segment: list of (n_ensemble, n_qpts, n_bands)
        freq_stack = _stack_frequencies(phonons)
        bs = mean_phonon.band_structure
        for seg_idx, (dist, mean_freq, conn) in enumerate(
            zip(bs.distances, bs.frequencies, bs.path_connections)
        ):
            seg_freqs = freq_stack[seg_idx]  # (n_ensemble, n_qpts, n_bands)
            mean_f = seg_freqs.mean(axis=0) * scale  # (n_qpts, n_bands)
            std_f = seg_freqs.std(axis=0) * scale  # (n_qpts, n_bands)
            for band_idx in range(mean_f.shape[1]):
                ax.fill_between(
                    dist,
                    mean_f[:, band_idx] - std_f[:, band_idx],
                    mean_f[:, band_idx] + std_f[:, band_idx],
                    color=color,
                    alpha=std_alpha,
                    lw=0,
                )
            ax.plot(dist, mean_f, color=color, lw=mean_linewidth)

    elif mode == "mean":
        _draw_band(ax, mean_phonon, scale, color=color, lw=mean_linewidth, alpha=1.0)

    elif mode == "spectral":
        _plot_spectral(ax, phonons, scale, cmap)

    _decorate_axes(ax, mean_phonon, scale, unit)
    ax.set_ylim(bottom=0)

    return fig, ax
