"""Exp 5: Descriptive panel of premium mean reversion."""
import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

from panel.ar1_estimation import (
    estimate_ar1,
    fit_panel_ar1,
    compute_halflife,
    compute_lambda_per_hour,
)
from panel.premium_series import (
    compute_basis,
    compute_relative_basis,
    build_premium_series,
)
from panel.funding_rule_docs import (
    get_funding_rule,
    classify_slope_sign,
    generate_symbol_summary_table,
)
from simulation.basis_ou import simulate_basis_ou


def run_exp5():
    print('=== EXP 5: Descriptive Panel — Premium Mean Reversion ===')
    results = {}

    # ------------------------------------------------------------------
    # 1. Basis computation demo
    # ------------------------------------------------------------------
    print('\n1. Basis computation (e_t = P_t - S_t, b_t = e_t / S_t)...')
    S = np.array([100.0, 100.3, 99.8, 100.1, 100.5])
    P = np.array([100.1, 100.4, 99.9, 100.2, 100.6])

    e = compute_basis(P, S)
    b = compute_relative_basis(P, S)
    print(f"   S     = {S}")
    print(f"   P     = {P}")
    print(f"   e_t   = {e}")
    print(f"   b_t   = {np.round(b, 6)}")
    print(f"   e_t == P_t - S_t: {np.allclose(e, P - S)}")
    results['basis_check'] = {
        'e_is_P_minus_S': bool(np.allclose(e, P - S)),
        'b_is_e_over_S': bool(np.allclose(b, e / S)),
        'b_range': (float(b.min()), float(b.max())),
    }

    # ------------------------------------------------------------------
    # 2. Relative basis range check on simulated data
    # ------------------------------------------------------------------
    print('\n2. Relative basis range on simulated OU paths...')
    ou_df = simulate_basis_ou(
        n_steps=5_000,
        phi_ar1=0.72,
        sigma_b=0.0006,
        seed=42,
    )
    b_sim = ou_df['b'].values
    b_in_typical_range = (b_sim >= -0.10).all() and (b_sim <= 0.10).all()
    print(f"   Simulated b_t range: [{b_sim.min():.4f}, {b_sim.max():.4f}]")
    print(f"   Within [-0.1, 0.1] : {b_in_typical_range}")
    results['relative_basis_range'] = {
        'b_min': float(b_sim.min()),
        'b_max': float(b_sim.max()),
        'within_10pct': b_in_typical_range,
    }

    # ------------------------------------------------------------------
    # 3. AR(1) half-life and lambda formulae
    # ------------------------------------------------------------------
    print('\n3. AR(1) half-life and lambda formulae...')
    phi_val = 0.72
    hl = compute_halflife(phi_val)
    lam = compute_lambda_per_hour(phi_val)
    hl_formula = -np.log(2) / np.log(phi_val)
    lam_formula = -np.log(phi_val)

    print(f"   phi = {phi_val}")
    print(f"   half-life = -ln(2)/ln(phi) = {hl:.4f} h  (formula: {hl_formula:.4f})")
    print(f"   lambda    = -ln(phi)        = {lam:.4f} /h  (formula: {lam_formula:.4f})")
    print(f"   Formula error (hl)  : {abs(hl - hl_formula):.2e}")
    print(f"   Formula error (lam) : {abs(lam - lam_formula):.2e}")
    results['halflife'] = {
        'phi': phi_val,
        'halflife_h': hl,
        'lambda_per_h': lam,
        'hl_formula_ok': abs(hl - hl_formula) < 1e-10,
        'lam_formula_ok': abs(lam - lam_formula) < 1e-10,
    }

    # ------------------------------------------------------------------
    # 4. AR(1) estimation on synthetic data with known phi
    # ------------------------------------------------------------------
    print('\n4. AR(1) estimation on synthetic data...')
    phi_target = 0.72
    ou_long = simulate_basis_ou(n_steps=10_000, phi_ar1=phi_target, sigma_b=0.0006, seed=7)
    ar1_est = estimate_ar1(ou_long['b'].values)
    print(f"   True phi       = {phi_target}")
    print(f"   Estimated phi  = {ar1_est['phi']:.4f}")
    print(f"   Estimation error = {abs(ar1_est['phi'] - phi_target):.4f}")
    print(f"   Estimated half-life = {ar1_est['halflife_hours']:.3f} h "
          f"(true = {compute_halflife(phi_target):.3f} h)")
    print(f"   R-squared      = {ar1_est['r_squared']:.4f}")

    results['ar1_estimation'] = {
        'phi_true': phi_target,
        'phi_estimated': ar1_est['phi'],
        'halflife_est_h': ar1_est['halflife_hours'],
        'estimation_ok': abs(ar1_est['phi'] - phi_target) < 0.05,
    }

    # ------------------------------------------------------------------
    # 5. Panel AR(1) fitting (synthetic multi-symbol panel)
    # ------------------------------------------------------------------
    print('\n5. Panel AR(1) fitting across synthetic symbols...')
    symbols = ['AAPL', 'AMZN', 'NVDA', 'GOOGL']
    phi_vals = [0.72, 0.68, 0.75, 0.70]
    panel_data = {}
    for sym, phi_true in zip(symbols, phi_vals):
        sim = simulate_basis_ou(n_steps=8_000, phi_ar1=phi_true, sigma_b=0.0006, seed=hash(sym) % 99999)
        panel_data[sym] = sim['b'].values

    panel_df = fit_panel_ar1(panel=panel_data)
    print(f"\n   {'Symbol':8s} {'phi_true':>10} {'phi_est':>10} {'hl_est':>10} {'error':>10}")
    for sym, phi_true in zip(symbols, phi_vals):
        row = panel_df.loc[panel_df['symbol'] == sym].iloc[0]
        print(f"   {sym:8s} {phi_true:10.4f} {row['phi']:10.4f} "
              f"{row['halflife_hours']:10.3f} {abs(row['phi']-phi_true):10.4f}")

    results['panel_ar1'] = panel_df.to_dict(orient='records')

    # ------------------------------------------------------------------
    # 6. Venue funding rule classification
    # ------------------------------------------------------------------
    print('\n6. Venue funding rule slope classification...')
    venues = ['hyperliquid', 'binance', 'okx']
    for v in venues:
        rule = get_funding_rule(v)
        sign_pos = classify_slope_sign(v, b=0.01)
        sign_neg = classify_slope_sign(v, b=-0.01)
        print(f"   {v:12s}: formula={rule['formula']:<30}  "
              f"sign(b>0)={sign_pos}  sign(b<0)={sign_neg}")
        results[f'slope_{v}'] = {'sign_positive_b': sign_pos, 'sign_negative_b': sign_neg}

    # Symbol summary table (build a synthetic universe for demonstration)
    universe_df = pd.DataFrame({
        'coin': symbols,
        'venue': ['hyperliquid'] * len(symbols),
        'listing_date_approx': ['2023-10'] * len(symbols),
    })
    summary_table = generate_symbol_summary_table(
        ar1_results=panel_df,
        universe=universe_df,
    )
    print(f"\n   Symbol summary table ({len(summary_table)} rows):")
    print(summary_table.to_string(index=False))
    summary_table.to_csv(RESULTS_DIR / 'table_symbol_summary.csv', index=False)

    # ------------------------------------------------------------------
    # 7. Plot: AR(1) phi distribution across synthetic panel
    # ------------------------------------------------------------------
    print('\n7. Plotting panel AR(1) phi distribution...')
    phis_est = [panel_df.loc[panel_df['symbol'] == sym, 'phi'].values[0] for sym in symbols]
    hls_est  = [panel_df.loc[panel_df['symbol'] == sym, 'halflife_hours'].values[0] for sym in symbols]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(symbols, phis_est, color='steelblue')
    axes[0].axhline(0.72, color='crimson', ls='--', lw=1.5, label='phi=0.72 target')
    axes[0].set_ylabel('AR(1) coefficient phi')
    axes[0].set_title('Estimated phi per symbol')
    axes[0].legend()

    axes[1].bar(symbols, hls_est, color='steelblue')
    axes[1].axhline(2.1, color='crimson', ls='--', lw=1.5, label='2.1 h target')
    axes[1].set_ylabel('Half-life (hours)')
    axes[1].set_title('Estimated half-life per symbol')
    axes[1].legend()

    fig.tight_layout()
    fig_path = RESULTS_DIR / 'fig_panel_ar1.png'
    fig.savefig(str(fig_path))
    plt.close(fig)
    print(f"   Saved: {fig_path}")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    out_path = RESULTS_DIR / 'exp5_results.txt'
    out_path.write_text(
        f"=== EXP 5: Descriptive Panel Results ===\n"
        f"Basis e_t = P_t - S_t: {results['basis_check']['e_is_P_minus_S']}\n"
        f"Simulated b_t range: [{results['relative_basis_range']['b_min']:.4f}, "
        f"{results['relative_basis_range']['b_max']:.4f}]\n"
        f"Half-life formula OK: {results['halflife']['hl_formula_ok']}\n"
        f"Lambda formula OK:    {results['halflife']['lam_formula_ok']}\n"
        f"AR(1) phi est error:  {abs(ar1_est['phi'] - phi_target):.4f}\n"
        f"Panel AR(1) symbols: {', '.join(panel_df['symbol'].tolist())}\n"
        f"Venue slopes:\n"
        + '\n'.join(f"  {v}: sign(b>0)={results[f'slope_{v}']['sign_positive_b']}"
                    for v in venues) + '\n'
    )
    print(f"\nSummary saved to {out_path}")

    return results


if __name__ == '__main__':
    run_exp5()
    print('\nExp 5 complete. Results saved to results/')
