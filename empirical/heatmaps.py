"""
Residual heatmap generation — Experiment 4 visualisations.

All figures are saved to /workspace/project/results/.  Callers may also
pass an explicit ``save_path`` to override the default file location.

Matplotlib is used with a non-interactive backend so figures can be
generated in headless environments.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_RESULTS_DIR = "/workspace/project/results"


def _ensure_results_dir() -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)


def _default_path(filename: str) -> str:
    _ensure_results_dir()
    return os.path.join(_RESULTS_DIR, filename)


def _hour_labels(timestamps: np.ndarray, event_idx: int) -> list[str]:
    """Build axis tick labels as hours relative to event (e.g. '-24h', '0', '+12h')."""
    if len(timestamps) == 0:
        return []
    t_event = timestamps[event_idx]
    labels = []
    for t in timestamps:
        delta_h = int(round((int(t) - int(t_event)) / 3_600_000))
        labels.append(f"{'+' if delta_h > 0 else ''}{delta_h}h")
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_residual_heatmap(
    R_matrix: np.ndarray,
    is_straddling: np.ndarray,
    timestamps: np.ndarray,
    event_idx: int,
    title: str = 'Residual Heatmap',
    save_path: Optional[str] = None,
) -> None:
    """Generate a residual heatmap over entry/exit pairs in the event window.

    The matrix cell (i, j) shows R(t_i, t_j).  The straddling region
    (i < event_idx <= j) is outlined with a red rectangle.  Upper-triangle
    cells (i >= j) are masked.

    Parameters
    ----------
    R_matrix : np.ndarray, shape (N, N)
        Pairwise residual matrix from :func:`~empirical.residuals.build_residual_matrix`.
    is_straddling : np.ndarray of bool, shape (N, N)
        Straddling indicator from :func:`~empirical.residuals.build_residual_matrix`.
    timestamps : np.ndarray of int
        Unix millisecond timestamps, length N.
    event_idx : int
        Index of the ex-dividend event in the window.
    title : str
        Figure title.
    save_path : str, optional
        Full file path for the saved figure.  Defaults to
        ``/workspace/project/results/<title>.png`` (spaces replaced with ``_``).
    """
    N = R_matrix.shape[0]
    labels = _hour_labels(timestamps, event_idx)
    tick_step = max(1, N // 12)
    tick_idx = list(range(0, N, tick_step))
    tick_labels = [labels[i] for i in tick_idx]

    # Mask upper triangle
    display = np.where(np.isnan(R_matrix), np.nan, R_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.nanpercentile(np.abs(display), 95)
    vmax = max(vmax, 1e-6)

    sns.heatmap(
        display,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Residual R(t₁, t₂)'},
    )

    # Overlay straddling rectangle
    # Straddling: row i in [0, event_idx-1], col j in [event_idx, N-1]
    if event_idx > 0 and event_idx < N:
        rect = mpatches.Rectangle(
            (event_idx, 0),           # (x=col_start, y=row_start)
            N - event_idx,             # width
            event_idx,                 # height
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            label='Straddling region',
        )
        ax.add_patch(rect)

    # Manual tick labels
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel('Exit time (t₂)')
    ax.set_ylabel('Entry time (t₁)')
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[rect] if event_idx > 0 and event_idx < N else [], loc='lower right')

    plt.tight_layout()
    if save_path is None:
        save_path = _default_path(title.replace(' ', '_') + '.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved residual heatmap to %s", save_path)


def plot_before_after_heatmaps(
    R_before: np.ndarray,
    R_after: np.ndarray,
    is_straddling: np.ndarray,
    timestamps: np.ndarray,
    event_idx: int,
    title_prefix: str = '',
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side residual heatmaps before and after counterfactual atom insertion.

    Uses a shared colour scale so the collapse of the straddling block is
    visually apparent.

    Parameters
    ----------
    R_before : np.ndarray, shape (N, N)
        Residual matrix before atom insertion.
    R_after : np.ndarray, shape (N, N)
        Residual matrix after counterfactual atom insertion.
    is_straddling : np.ndarray of bool, shape (N, N)
        Straddling indicator.
    timestamps : np.ndarray of int
        Unix millisecond timestamps.
    event_idx : int
        Ex-dividend event index.
    title_prefix : str
        Prefix prepended to each subplot title.
    save_path : str, optional
        Save path override.
    """
    N = R_before.shape[0]
    labels = _hour_labels(timestamps, event_idx)
    tick_step = max(1, N // 12)
    tick_idx = list(range(0, N, tick_step))
    tick_labels = [labels[i] for i in tick_idx]

    combined = np.concatenate([
        R_before[~np.isnan(R_before)],
        R_after[~np.isnan(R_after)],
    ])
    vmax = float(np.percentile(np.abs(combined), 95)) if len(combined) > 0 else 1e-4
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    for ax, R, subtitle in zip(axes, (R_before, R_after), ('Before (no atom)', 'After (with atom)')):
        sns.heatmap(
            R,
            ax=ax,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Residual'},
        )
        if event_idx > 0 and event_idx < N:
            rect = mpatches.Rectangle(
                (event_idx, 0),
                N - event_idx,
                event_idx,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
            )
            ax.add_patch(rect)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel('Exit time (t₂)')
        ax.set_ylabel('Entry time (t₁)')
        full_title = f"{title_prefix} — {subtitle}" if title_prefix else subtitle
        ax.set_title(full_title, fontsize=11)

    plt.suptitle(
        f"{title_prefix} Counterfactual Atom Insertion" if title_prefix
        else "Counterfactual Atom Insertion",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    if save_path is None:
        prefix_slug = title_prefix.replace(' ', '_') if title_prefix else 'before_after'
        save_path = _default_path(f"{prefix_slug}_before_after.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved before/after heatmaps to %s", save_path)


def plot_cross_venue_comparison(
    results_by_venue: dict,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing straddling residual means across venues.

    Parameters
    ----------
    results_by_venue : dict
        Mapping of ``{venue_name: {'R_matrix': np.ndarray, 'stats': dict}}``.
        ``stats`` must contain at minimum ``straddling_mean`` and
        ``non_straddling_mean`` keys (output of
        :func:`~empirical.residuals.compute_straddling_stats`).
    save_path : str, optional
        Save path override.
    """
    if not results_by_venue:
        logger.warning("plot_cross_venue_comparison: no data provided.")
        return

    venues = list(results_by_venue.keys())
    strad_means = [results_by_venue[v]['stats'].get('straddling_mean', float('nan'))
                   for v in venues]
    non_strad_means = [results_by_venue[v]['stats'].get('non_straddling_mean', float('nan'))
                       for v in venues]

    x = np.arange(len(venues))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(venues) * 2), 5))
    bars1 = ax.bar(x - width / 2, strad_means, width, label='Straddling', color='tomato')
    bars2 = ax.bar(x + width / 2, non_strad_means, width, label='Non-straddling', color='steelblue')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(venues, fontsize=10)
    ax.set_ylabel('Mean Residual R')
    ax.set_title('Cross-Venue Straddling Residual Comparison')
    ax.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = _default_path('cross_venue_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved cross-venue comparison to %s", save_path)
