"""
All paper figures for the equity perpetual fairness study.

Figures saved to /workspace/project/results/.

Functions
---------
plot_dividend_path_comparison     : 3 cumulative wealth paths around ex-dividend event.
plot_funding_rule_comparison      : cumulative funding under deployed vs. fair rules.
plot_funding_slope_comparison     : f vs. b scatter showing opposite slope signs.
plot_discrepancy_illustration     : interval-discrepancy for AC vs. Dirac measure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("/workspace/project/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Common style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to save_path or to RESULTS_DIR if save_path is None."""
    if save_path is None:
        fname = Path(plt.gcf().get_label() or "figure").with_suffix(".png")
        save_path = str(RESULTS_DIR / fname)
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Dividend path comparison
# ---------------------------------------------------------------------------

def plot_dividend_path_comparison(
    df: pd.DataFrame,
    tau: int,
    D: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot three cumulative wealth paths around an ex-dividend event.

    Panels:
      Left  — spot price S_t with ex-dividend drop at tau.
      Right — cumulative P&L for (1) physical shareholder, (2) standard perp long,
              (3) event-credit perp long.

    Parameters
    ----------
    df        : output of simulate_dividend_paths (must contain the required columns).
    tau       : ex-dividend step index.
    D         : discrete dividend size (for annotation).
    save_path : file path for the saved figure; defaults to results/fig_dividend_paths.png.
    """
    t = df["t"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.set_label("fig_dividend_paths")

    # --- Left panel: spot price -------------------------------------------
    ax = axes[0]
    ax.plot(t, df["S"].values, color="steelblue", lw=1.8, label="Spot $S_t$")
    ax.axvline(tau, color="crimson", lw=1.4, ls="--", label=f"Ex-div ($\\tau={tau}$)")
    ax.annotate(
        f"$\\Delta S_{{\\tau}} = -D$",
        xy=(tau, df["S"].iloc[tau]),
        xytext=(tau + 3, df["S"].iloc[tau] + D * 2.5),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        color="crimson", fontsize=10,
    )
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Spot price")
    ax.set_title("Spot price around ex-dividend")
    ax.legend(fontsize=9)

    # --- Right panel: cumulative wealth paths --------------------------------
    ax = axes[1]
    ax.plot(t, df["shareholder_wealth"].values,
            color="steelblue", lw=2.0, label="Physical shareholder")
    ax.plot(t, df["standard_perp_wealth"].values,
            color="tomato", lw=1.8, ls="--", label="Standard perp long (no credit)")
    ax.plot(t, df["event_credit_wealth"].values,
            color="forestgreen", lw=1.8, ls=":", label="Event-credit perp long")
    ax.axvline(tau, color="gray", lw=1.2, ls="--", alpha=0.7)

    # Annotate the deficit D
    t_ann   = int(len(t) * 0.85)
    sw_ann  = float(df["shareholder_wealth"].iloc[t_ann])
    spw_ann = float(df["standard_perp_wealth"].iloc[t_ann])
    ax.annotate(
        "",
        xy=(t_ann, spw_ann),
        xytext=(t_ann, sw_ann),
        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5),
    )
    ax.text(
        t_ann + 1, (sw_ann + spw_ann) / 2,
        f"$D \\approx {D:.3f}$",
        color="crimson", fontsize=10, va="center",
    )
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title("Wealth paths: shareholder vs. perp long")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Ex-dividend event: shareholder vs. perpetual long", fontsize=13, y=1.01)
    fig.tight_layout()

    out = save_path or str(RESULTS_DIR / "fig_dividend_paths.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Cumulative funding rule comparison
# ---------------------------------------------------------------------------

def plot_funding_rule_comparison(
    b_sim: np.ndarray,
    f_deploy: np.ndarray,
    f_fair: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot cumulative funding under the deployed and fair funding rules.

    Shows both the pointwise funding rates and their cumulative sums over a
    simulated basis path, highlighting the divergence.

    Parameters
    ----------
    b_sim    : simulated basis fraction path, shape (T,) or (T+1,).
    f_deploy : deployed funding rates, shape (T,).
    f_fair   : fair funding rates, shape (T,).
    save_path: file path; defaults to results/fig_funding_rule_comparison.png.
    """
    T = len(f_deploy)
    t = np.arange(T)

    cum_deploy = np.cumsum(f_deploy)
    cum_fair   = np.cumsum(f_fair)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.set_label("fig_funding_rule_comparison")

    # --- Panel 1: basis path ------------------------------------------------
    ax = axes[0]
    ax.plot(t, b_sim[:T], color="steelblue", lw=1.4)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Basis fraction $b_t$")
    ax.set_title("Simulated OU basis $b_t$")

    # --- Panel 2: funding rates ---------------------------------------------
    ax = axes[1]
    ax.plot(t, f_deploy, color="tomato", lw=1.4, label="Deployed $f_{\\mathrm{deploy}}$")
    ax.plot(t, f_fair,   color="steelblue", lw=1.4, ls="--", label="Fair $f_{\\mathrm{fair}}$")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Funding rate")
    ax.set_title("Funding rates along basis path")
    ax.legend(fontsize=9)

    # --- Panel 3: cumulative funding ----------------------------------------
    ax = axes[2]
    ax.plot(t, cum_deploy, color="tomato", lw=1.8, label="Cumulative deployed")
    ax.plot(t, cum_fair,   color="steelblue", lw=1.8, ls="--", label="Cumulative fair")
    ax.fill_between(t, cum_deploy, cum_fair,
                    where=(cum_deploy > cum_fair), alpha=0.2, color="tomato",
                    label="Deploy > Fair")
    ax.fill_between(t, cum_deploy, cum_fair,
                    where=(cum_deploy <= cum_fair), alpha=0.2, color="steelblue",
                    label="Fair > Deploy")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative funding (per unit P)")
    ax.set_title("Cumulative funding divergence")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Deployed vs. fair funding rules along OU basis path", fontsize=13, y=1.02)
    fig.tight_layout()

    out = save_path or str(RESULTS_DIR / "fig_funding_rule_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Slope comparison scatter
# ---------------------------------------------------------------------------

def plot_funding_slope_comparison(
    b_grid: np.ndarray,
    f_deploy: np.ndarray,
    f_fair: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Scatter plot of the deployed and fair funding rates vs. the basis b.

    Shows that the two rules have opposite slopes: the deployed rule
    increases with premium (kappa > 0), while the fair rule decreases.

    Parameters
    ----------
    b_grid   : basis fraction values, shape (K,).
    f_deploy : deployed funding rates evaluated on b_grid, shape (K,).
    f_fair   : fair funding rates evaluated on b_grid, shape (K,).
    save_path: file path; defaults to results/fig_funding_slope_comparison.png.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.set_label("fig_funding_slope_comparison")

    ax.plot(b_grid, f_deploy, color="tomato", lw=2.2, label="Deployed: $f = (r{-}q) + \\kappa b$")
    ax.plot(b_grid, f_fair,   color="steelblue", lw=2.2, ls="--",
            label="Fair (eq. 8): negative slope")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.axvline(0, color="black", lw=0.8, ls=":")

    # Annotate slope directions
    mid = len(b_grid) // 2
    ax.annotate(
        "Deployed: positive slope",
        xy=(b_grid[mid + 2], f_deploy[mid + 2]),
        xytext=(b_grid[mid + 2], f_deploy[mid + 2] + (f_deploy[-1] - f_deploy[0]) * 0.15),
        color="tomato", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="tomato"),
    )
    ax.annotate(
        "Fair: negative slope",
        xy=(b_grid[mid - 2], f_fair[mid - 2]),
        xytext=(b_grid[mid - 2], f_fair[mid - 2] - (f_deploy[-1] - f_deploy[0]) * 0.15),
        color="steelblue", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="steelblue"),
    )

    ax.set_xlabel("Basis fraction $b_t = (P_t - S_t)/S_t$")
    ax.set_ylabel("Funding rate $f(b_t)$")
    ax.set_title("Deployed vs. fair funding: opposite slope signs")
    ax.legend(fontsize=10)

    fig.tight_layout()
    out = save_path or str(RESULTS_DIR / "fig_funding_slope_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Interval-discrepancy illustration
# ---------------------------------------------------------------------------

def plot_discrepancy_illustration(
    tau: float,
    D: float,
    horizon: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Illustrate the interval-discrepancy metric for AC measure vs. Dirac atom.

    Left panel:  Dirac mass nu = D*delta_tau vs. best Gaussian AC approximation.
    Right panel: |mu((tau-eps, tau]) - D| vs. eps, showing convergence to D.

    Parameters
    ----------
    tau      : ex-dividend time (Dirac atom location).
    D        : Dirac mass size.
    horizon  : right endpoint of the time domain.
    save_path: file path; defaults to results/fig_discrepancy_illustration.png.
    """
    from scipy.stats import norm

    t_fine = np.linspace(0.0, horizon, 500)

    # Best Gaussian approximations: place mass just to the right of tau
    sigma_values = [0.3, 0.1, 0.03]
    colors       = ["#5e9cc5", "#2171b5", "#08306b"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.set_label("fig_discrepancy_illustration")

    # --- Left panel: densities -----------------------------------------------
    ax = axes[0]
    for sigma, col in zip(sigma_values, colors):
        mu_loc  = tau + sigma
        density = D * norm.pdf(t_fine, mu_loc, sigma)
        ax.plot(t_fine, density, color=col, lw=1.8,
                label=f"$\\mathcal{{N}}(\\tau+{sigma}, {sigma}^2)$ · D")

    # Dirac mass: draw an arrow
    ax.annotate(
        "", xy=(tau, D * 1.05), xytext=(tau, 0),
        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5),
    )
    ax.text(tau + 0.01 * horizon, D * 0.55, f"$D \\cdot \\delta_{{\\tau}}$",
            color="crimson", fontsize=11)

    ax.axvline(tau, color="crimson", lw=1.2, ls="--", alpha=0.6)
    ax.set_xlim(0, horizon)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Density / mass")
    ax.set_title("AC approximations to $\\nu = D\\,\\delta_\\tau$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)

    # --- Right panel: discrepancy vs. eps ------------------------------------
    ax = axes[1]
    eps_vals = np.logspace(-3, np.log10(horizon / 2), 60)

    for sigma, col in zip(sigma_values, colors):
        mu_loc = tau + sigma
        disc   = np.array([
            abs(D * (norm.cdf(tau, mu_loc, sigma) - norm.cdf(tau - eps, mu_loc, sigma)) - D)
            for eps in eps_vals
        ])
        ax.semilogx(eps_vals, disc, color=col, lw=1.8,
                    label=f"$\\sigma={sigma}$")

    ax.axhline(D, color="crimson", lw=1.8, ls="--",
               label=f"Lower bound $D={D}$")

    ax.set_xlabel("Interval half-width $\\varepsilon$")
    ax.set_ylabel("$|\\mu((\\tau{-}\\varepsilon,\\tau]) - D|$")
    ax.set_title("Discrepancy contribution $\\to D$ as $\\varepsilon \\to 0$")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Interval-discrepancy: any AC measure has $\\Delta(\\mu,\\nu) \\geq D$",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    out = save_path or str(RESULTS_DIR / "fig_discrepancy_illustration.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Convenience runner: produce all figures
# ---------------------------------------------------------------------------

def produce_all_figures() -> None:
    """Generate and save all four paper figures using default parameters."""
    from simulation.dividend_paths import simulate_dividend_paths
    from simulation.basis_ou import simulate_basis_ou
    from simulation.funding_rules import (
        deployed_funding_rule,
        fair_funding_rule,
        slope_comparison,
    )

    r, q   = 0.05 / 252 / 24, 0.02 / 252 / 24
    lam    = np.log(2) / 2.1
    sig_b  = 0.0006
    sig_S  = 0.20 / (252 * 24) ** 0.5
    rho    = 0.0
    kappa  = 0.05

    # Figure 1
    df = simulate_dividend_paths(D_frac=0.01, tau=36, n_steps=72, seed=42)
    D_val = float((df.loc[df["t"] >= 36, "shareholder_wealth"]
                   - df.loc[df["t"] >= 36, "standard_perp_wealth"]).iloc[0])
    plot_dividend_path_comparison(df, tau=36, D=D_val)

    # Figures 2 & 3 — use OU basis simulation
    df_ou  = simulate_basis_ou(n_steps=72, phi_ar1=0.72, sigma_b=sig_b, seed=42)
    b_path = df_ou["b"].values[:-1]     # shape (72,)
    P_path = (df_ou["S"].values[:-1] * (1.0 + b_path))  # approx perp mark

    f_dep  = deployed_funding_rule(b_path, r, q, kappa)
    f_fair = fair_funding_rule(b_path, r, q, lam, sig_b, sig_S, rho)
    plot_funding_rule_comparison(df_ou["b"].values, f_dep, f_fair)

    b_grid  = np.linspace(-0.05, 0.05, 200)
    f_dep_g = deployed_funding_rule(b_grid, r, q, kappa)
    f_fair_g = fair_funding_rule(b_grid, r, q, lam, sig_b, sig_S, rho)
    plot_funding_slope_comparison(b_grid, f_dep_g, f_fair_g)

    # Figure 4
    plot_discrepancy_illustration(tau=0.5, D=1.0, horizon=1.0)

    print("\nAll figures saved to", RESULTS_DIR)


if __name__ == "__main__":
    produce_all_figures()
