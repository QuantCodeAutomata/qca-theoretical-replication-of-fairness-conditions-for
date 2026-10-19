"""Exp 2: Simulation replication of long-perp dividend loss."""
import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

from simulation.dividend_paths import simulate_dividend_paths, verify_dividend_loss
from simulation.basis_ou import simulate_basis_ou, estimate_ar1_persistence, compute_halflife
from simulation.funding_rules import (
    deployed_funding_rule, fair_funding_rule, fair_funding_small_b_approx,
    cumulative_funding, compare_funding_rules_pointwise,
)
from simulation.figures import (
    plot_dividend_path_comparison,
    plot_funding_rule_comparison,
    plot_funding_slope_comparison,
)


def run_exp2():
    print('=== EXP 2: Simulation Replication ===')
    results = {}

    # Parameters
    r_h  = 0.05 / 252 / 24   # hourly risk-free rate
    q_h  = 0.02 / 252 / 24   # hourly continuous yield
    lam  = np.log(2) / 2.1   # OU mean-reversion (half-life ≈ 2.1 h)
    phi  = np.exp(-lam)       # ≈ 0.72 per hour
    kap  = 0.05               # deployed premium-index slope
    sig_b = 0.0006            # basis vol per hour
    sig_S = 0.20 / (252 * 24) ** 0.5
    tau  = 36
    n_steps = 72

    # ------------------------------------------------------------------
    # 1. Dividend path comparison
    # ------------------------------------------------------------------
    print('\n1. Simulating dividend paths around ex-dividend event...')
    df = simulate_dividend_paths(
        S0=100.0,
        r=r_h,
        q=q_h,
        sigma_S=sig_S,
        D_frac=0.01,
        tau=tau,
        n_steps=n_steps,
        seed=42,
    )

    # Infer D from simulation output
    D_sim = float(
        (df.loc[df['t'] == tau, 'event_credit_wealth']
         - df.loc[df['t'] == tau, 'standard_perp_wealth']).iloc[0]
    )
    print(f"   Simulated dividend D ≈ {D_sim:.4f}")
    print(f"   Final shareholder wealth      : {df['shareholder_wealth'].iloc[-1]:.6f}")
    print(f"   Final standard perp wealth    : {df['standard_perp_wealth'].iloc[-1]:.6f}")
    print(f"   Final event-credit wealth     : {df['event_credit_wealth'].iloc[-1]:.6f}")
    print(f"   Standard perp trailing gap    : "
          f"{df['shareholder_wealth'].iloc[-1] - df['standard_perp_wealth'].iloc[-1]:.6f}")

    chk = verify_dividend_loss(df, D=D_sim, tau=tau)
    results['dividend_loss'] = {
        'D': D_sim,
        'standard_deficit_ok': chk['standard_deficit_ok'],
        'event_credit_matches': chk['event_credit_matches'],
    }
    print(f"   Standard deficit ≈ D : {chk['standard_deficit_ok']}")
    print(f"   Event-credit matches : {chk['event_credit_matches']}")

    plot_dividend_path_comparison(
        df=df, tau=tau, D=D_sim,
        save_path=str(RESULTS_DIR / 'fig_dividend_paths.png'),
    )

    # ------------------------------------------------------------------
    # 2. Basis OU simulation and AR(1) estimation
    # ------------------------------------------------------------------
    print('\n2. Simulating mean-reverting basis (AR(1) / OU)...')
    sim_df = simulate_basis_ou(
        n_steps=5_000,
        phi_ar1=phi,
        sigma_b=sig_b,
        sigma_S=sig_S,
        rho=0.0,
        b0=0.0,
        S0=100.0,
        r=r_h,
        q=q_h,
        seed=42,
    )
    b_series = sim_df['b'].values
    phi_hat = estimate_ar1_persistence(b_series)
    hl_hat  = compute_halflife(phi_hat)
    hl_true = compute_halflife(phi)

    print(f"   Target phi             = {phi:.4f}  (half-life = {hl_true:.3f} h)")
    print(f"   Estimated phi          = {phi_hat:.4f}  (half-life = {hl_hat:.3f} h)")
    print(f"   Estimation error       = {abs(phi_hat - phi):.4f}")

    lam_est = -np.log(phi_hat)
    print(f"   Estimated lambda       = {lam_est:.4f} /hr  (target {lam:.4f})")

    results['basis_ou'] = {
        'phi_true': phi,
        'phi_estimated': phi_hat,
        'halflife_true_h': hl_true,
        'halflife_est_h': hl_hat,
        'lambda_estimated': lam_est,
        'estimation_ok': abs(phi_hat - phi) < 0.10,
    }

    # ------------------------------------------------------------------
    # 3. Funding rule comparison
    # ------------------------------------------------------------------
    print('\n3. Comparing deployed vs fair funding rules...')
    b_path = sim_df['b'].values[:n_steps]
    P_path = sim_df['S'].values[:n_steps] * (1 + b_path)   # P = S*(1+b)

    f_dep  = deployed_funding_rule(b_path, r=r_h, q=q_h, kappa=kap)
    f_fair = fair_funding_rule(b_path, r=r_h, q=q_h, lam=lam,
                               sigma_b=sig_b, sigma_S=sig_S, rho=0.0)
    f_approx = fair_funding_small_b_approx(b_path, r=r_h, q=q_h, lam=lam)

    cum_dep  = cumulative_funding(f_dep,  P_path, dt=1.0)
    cum_fair = cumulative_funding(f_fair, P_path, dt=1.0)

    print(f"   Mean deployed funding  = {f_dep.mean():.6f}")
    print(f"   Mean fair funding      = {f_fair.mean():.6f}")
    print(f"   Mean approx error      = {np.abs(f_fair - f_approx).mean():.2e}")
    print(f"   Final cum. deployed    = {cum_dep[-1]:.6f}")
    print(f"   Final cum. fair        = {cum_fair[-1]:.6f}")

    # Verify slope signs
    b_grid = np.linspace(-0.05, 0.05, 201)
    fd = deployed_funding_rule(b_grid, r=r_h, q=q_h, kappa=kap)
    ff = fair_funding_rule(b_grid, r=r_h, q=q_h, lam=lam, sigma_b=sig_b,
                            sigma_S=sig_S, rho=0.0)
    slope_deploy = (fd[-1] - fd[0]) / (b_grid[-1] - b_grid[0])
    slope_fair   = (ff[-1] - ff[0]) / (b_grid[-1] - b_grid[0])
    print(f"\n   Deployed slope df/db = {slope_deploy:.4f}  (positive)")
    print(f"   Fair slope df/db     = {slope_fair:.4f}  (negative)")

    results['funding_comparison'] = {
        'slope_deploy': float(slope_deploy),
        'slope_fair': float(slope_fair),
        'deploy_slope_positive': slope_deploy > 0,
        'fair_slope_negative': slope_fair < 0,
        'mean_f_deploy': float(f_dep.mean()),
        'mean_f_fair': float(f_fair.mean()),
    }

    # Save figures
    plot_funding_rule_comparison(
        b_sim=b_path,
        f_deploy=f_dep,
        f_fair=f_fair,
        save_path=str(RESULTS_DIR / 'fig_funding_rule_comparison.png'),
    )
    plot_funding_slope_comparison(
        b_grid=b_grid,
        f_deploy=fd,
        f_fair=ff,
        save_path=str(RESULTS_DIR / 'fig_funding_slope_comparison.png'),
    )

    # ------------------------------------------------------------------
    # Save results summary
    # ------------------------------------------------------------------
    summary_lines = [
        "=== EXP 2: Simulation Results ===",
        f"Discrete dividend D = {D_sim:.4f}",
        f"Standard deficit ≈ D: {chk['standard_deficit_ok']}",
        f"Event-credit matches shareholder: {chk['event_credit_matches']}",
        f"AR(1) phi_true={phi:.4f}, phi_hat={phi_hat:.4f}, "
        f"error={abs(phi_hat-phi):.4f}",
        f"Half-life true={hl_true:.3f} h, estimated={hl_hat:.3f} h",
        f"Deployed slope = {slope_deploy:.4f} (positive)",
        f"Fair slope = {slope_fair:.4f} (negative)",
    ]
    out_path = RESULTS_DIR / 'exp2_results.txt'
    out_path.write_text('\n'.join(summary_lines))
    print(f"\nSummary saved to {out_path}")

    return results


if __name__ == '__main__':
    results = run_exp2()
    print('\nExp 2 complete. Results saved to results/')
