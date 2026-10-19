"""Exp 1: Theoretical replication of fairness conditions."""
import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

from theory.identities import (
    verify_residual_identity,
    verify_basis_substituted_form,
    compute_fair_funding_target,
    demonstrate_weak_fairness,
    demonstrate_strong_fairness as demonstrate_strong_fairness_identity,
)
from theory.jumps import (
    demonstrate_weak_fairness_condition,
    demonstrate_strong_fairness,
    counterexample_no_atom,
)
from theory.discrepancy import (
    compute_discrepancy,
    compute_discrepancy_fast,
    minimize_discrepancy,
)
from theory.ou_fair_rule import (
    compute_fair_funding_rule,
    compare_fair_vs_deployed,
    projection_benchmark,
    projection_nonneg,
    symbolic_derive_fair_funding,
)
from simulation.figures import (
    plot_discrepancy_illustration,
    plot_funding_slope_comparison,
)


def run_exp1():
    print('=== EXP 1: Theoretical Replication ===')
    results = {}

    # ------------------------------------------------------------------
    # 1. Residual accounting identity: R = M_{tau2} - M_{tau1}
    # ------------------------------------------------------------------
    print('\n1. Verifying residual accounting identity...')
    identity_result = verify_residual_identity(n_steps=1000, seed=42)
    results['identity'] = identity_result
    print(f"   R               = {identity_result['R']:.6e}")
    print(f"   M increment     = {identity_result['M_increment']:.6e}")
    print(f"   Identity error  = {identity_result['identity_error']:.2e}")
    print(f"   Identity holds  : {identity_result['identity_holds']}")

    # Basis-substituted form (symbolic)
    sym_result = verify_basis_substituted_form()
    results['symbolic_identity'] = {
        'identity_holds': sym_result['identity_holds'],
        'R_equals_M_diff': sym_result['R_equals_M_diff'],
    }
    print(f"   Symbolic identity holds : {sym_result['identity_holds']}")
    print(f"   R equals M diff         : {sym_result['R_equals_M_diff']}")

    # Fair funding target Y_t = (r*S + mu_e - a) / P
    r, S, mu_e, a, P = 0.05 / 8760, 100.0, 0.5 / 8760 * 100, 0.02 / 8760 * 100, 100.5
    Y = compute_fair_funding_target(r, S, mu_e, a, P)
    Y_check = (r * S + mu_e - a) / P
    results['fair_funding_target'] = {'Y_t': float(Y), 'formula_error': abs(float(Y) - Y_check)}
    print(f"   Y_t (fair funding target) = {float(Y):.6e}")
    print(f"   Formula error             = {abs(float(Y) - Y_check):.2e}")

    # ------------------------------------------------------------------
    # 2. Weak fairness condition (martingale view)
    # ------------------------------------------------------------------
    print('\n2. Demonstrating weak fairness condition (martingale)...')
    wf_identity = demonstrate_weak_fairness(n_steps=2000, seed=42, n_bootstrap=1000)
    results['weak_fairness_identity'] = {
        'identity_holds': wf_identity['identity_holds'],
        'wf_passes': wf_identity['weak_fairness_check']['passes'],
        'mean_increment': wf_identity['weak_fairness_check']['mean_increment'],
    }
    print(f"   Weak fairness passes  : {wf_identity['weak_fairness_check']['passes']}")
    print(f"   Mean M increment      : {wf_identity['weak_fairness_check']['mean_increment']:.4e}")

    # Weak fairness condition (jump view): E[Delta_e_tau] = D
    print('\n2b. Demonstrating weak fairness jump condition (E[Delta_e] = D)...')
    wf_jump = demonstrate_weak_fairness_condition(D=1.0, S0=100.0, tau=36,
                                                  n_steps=72, n_paths=20_000, seed=42)
    results['weak_fairness_jump'] = {
        'E_Delta_e': wf_jump['E_Delta_e'],
        'D': wf_jump['D'],
        'condition_met': wf_jump['condition_met'],
    }
    print(f"   E[Delta_e_tau] = {wf_jump['E_Delta_e']:.6f}  (target D = {wf_jump['D']:.6f})")
    print(f"   Condition met  : {wf_jump['condition_met']}")

    # Strong fairness (jump view)
    sf_jump = demonstrate_strong_fairness(D=1.0, S0=100.0, tau=36, n_steps=72, seed=42)
    results['strong_fairness_jump'] = {
        'Delta_e_tau': sf_jump['Delta_e_tau'],
        'strong_fairness_ok': sf_jump['strong_fairness_ok'],
    }
    print(f"   Strong fairness (jump): Delta_e_tau = {sf_jump['Delta_e_tau']:.4f}  "
          f"OK={sf_jump['strong_fairness_ok']}")

    # Strong fairness (path view)
    sf_path = demonstrate_strong_fairness_identity(n_steps=2000, seed=42)
    results['strong_fairness_path'] = {
        'sf_passes': sf_path['strong_fairness_check']['passes'],
        'max_deviation': sf_path['strong_fairness_check']['max_deviation'],
    }
    print(f"   Strong fairness (path): max|R| = "
          f"{sf_path['strong_fairness_check']['max_deviation']:.2e}  "
          f"passes={sf_path['strong_fairness_check']['passes']}")

    # Counterexample: standard perp cannot achieve fairness continuously
    ce = counterexample_no_atom(D=1.0, S0=100.0, tau=36, n_steps=72, seed=42)
    results['counterexample'] = {
        'delta_e_tau': ce['Delta_e_tau'],
        'deficit': ce['deficit'],
        'R_deficit_post': ce['R_deficit_post'],
    }
    print(f"\n2c. Counterexample (no atom):")
    print(f"   Delta_e_tau (standard) = {ce['Delta_e_tau']:.4f}  (should be 0)")
    print(f"   Permanent deficit      = {ce['deficit']:.4f}  (= D)")
    print(f"   Post-tau R deficit     = {ce['R_deficit_post']:.4f}")

    # ------------------------------------------------------------------
    # 3. Interval-discrepancy metric Delta(mu, nu) >= D
    # ------------------------------------------------------------------
    print('\n3. Computing interval-discrepancy metric...')
    tau_d, D_d, horizon = 0.5, 1.0, 1.0
    grid = np.linspace(0, horizon, 150)
    unif_density = lambda t: D_d / horizon if 0 <= t <= horizon else 0.0
    # Fast analytical computation — exploits problem structure so the
    # shrinking-interval case (a→tau-, b=tau) is computed at the limit.
    disc_fast = compute_discrepancy_fast(unif_density, tau_d, D_d, grid)
    disc = disc_fast['discrepancy']
    # The fast version analytically captures the shrinking-interval limit, so
    # disc should be exactly D for any absolutely continuous mu.
    lower_bound_holds = disc >= D_d - 1e-6
    results['discrepancy'] = {
        'uniform_density_disc': float(disc),
        'lower_bound_D': D_d,
        'lower_bound_holds': lower_bound_holds,
    }
    print(f"   Uniform density discrepancy = {disc:.4f}  (lower bound D = {D_d})")
    print(f"   Lower bound holds           : {lower_bound_holds}")

    # Minimisation search
    print('   Running discrepancy minimisation...')
    opt = minimize_discrepancy(tau=tau_d, D=D_d, horizon=horizon, n_grid=100)
    results['discrepancy_opt'] = {
        'min_discrepancy': opt['min_discrepancy'],
        'lower_bound': opt['lower_bound'],
        'gap': opt['gap'],
    }
    print(f"   Min discrepancy (over Gaussian family) = {opt['min_discrepancy']:.4f}")
    print(f"   Gap above D                            = {opt['gap']:.4f}")

    # Save discrepancy illustration figure
    plot_discrepancy_illustration(tau=tau_d, D=D_d, horizon=horizon,
                                  save_path=str(RESULTS_DIR / 'fig_discrepancy_illustration.png'))

    # ------------------------------------------------------------------
    # 4. OU-based fair funding rule (Eq. 8)
    # ------------------------------------------------------------------
    print('\n4. OU-based fair funding rule (Eq. 8)...')
    r_h = 0.05 / 8760
    q_h = 0.02 / 8760
    lam = 0.329          # per hour, corresponding to half-life ≈ 2.1 h

    # Symbolic derivation check
    sym_ou = symbolic_derive_fair_funding()
    print(f"   Symbolic derivation equiv: {sym_ou['derivation_equiv']}")

    print(f"   Parameters: r={r_h:.6f}/hr, q={q_h:.6f}/hr, lambda={lam:.4f}/hr")
    b_values = [-0.02, -0.01, 0.0, 0.01, 0.02]
    ou_table = []
    for b_val in b_values:
        f_exact = compute_fair_funding_rule(r=r_h, q=q_h, lambda_ou=lam, b=b_val)
        f_approx = (r_h - q_h) - lam * b_val
        f_nonneg = projection_nonneg(f_exact)  # non-negative clip for scalar
        ou_table.append({'b': b_val, 'f_exact': f_exact, 'f_approx': f_approx,
                         'approx_error': abs(f_exact - f_approx), 'f_clip': f_nonneg})
        print(f"   b={b_val:+.3f}: f_exact={f_exact:.6f}, "
              f"f_approx={f_approx:.6f}, f_clip={f_nonneg:.6f}")

    results['ou_fair_rule'] = ou_table

    # Kernel projection benchmark: f*(x) using simulated samples
    print('   Computing projection benchmark f*(x) via Nadaraya-Watson...')
    np.random.seed(0)
    n_sim = 5000
    b_sim = np.random.normal(0, 0.0006, n_sim)
    P_sim = 100.0 * (1 + b_sim)
    S_sim = 100.0 * np.ones(n_sim)
    Y_sim = (r_h * S_sim - q_h * S_sim - lam * b_sim * S_sim) / P_sim
    x_grid_small = np.linspace(-0.003, 0.003, 15)
    e_sim = b_sim * S_sim
    f_star_vals = projection_benchmark(Y_sim, P_sim, e_sim, x_grid_small)
    results['projection_benchmark'] = {'x_grid': x_grid_small.tolist(),
                                        'f_star': f_star_vals.tolist()}
    print(f"   f*(e=0) ≈ {float(np.nanmean(f_star_vals)):.6f}")

    # ------------------------------------------------------------------
    # 5. Compare fair vs deployed funding rules
    # ------------------------------------------------------------------
    print('\n5. Comparing fair vs deployed funding rules...')
    b_grid = np.linspace(-0.05, 0.05, 201)
    comp = compare_fair_vs_deployed(b_grid=b_grid, r=r_h, q=q_h, lam=lam,
                                     rho=0.0, kappa=0.05)
    slope_sign_fair = 'negative' if float(np.nanmean(comp['slope_fair'])) < 0 else 'positive'
    slope_sign_deploy = 'positive' if float(np.nanmean(comp['slope_deploy'])) > 0 else 'negative'
    results['comparison'] = {
        'slope_sign_fair': slope_sign_fair,
        'slope_sign_deploy': slope_sign_deploy,
        'f_fair_at_zero': float(comp['f_fair'][len(b_grid) // 2]),
        'f_deploy_at_zero': float(comp['f_deploy'][len(b_grid) // 2]),
    }
    print(f"   Fair rule slope sign    : {slope_sign_fair}")
    print(f"   Deployed rule slope sign: {slope_sign_deploy}")
    print(f"   f_fair(b=0)  = {results['comparison']['f_fair_at_zero']:.6f}")
    print(f"   f_deploy(b=0)= {results['comparison']['f_deploy_at_zero']:.6f}")

    # Save slope comparison figure
    plot_funding_slope_comparison(
        b_grid=b_grid,
        f_deploy=comp['f_deploy'],
        f_fair=comp['f_fair'],
        save_path=str(RESULTS_DIR / 'fig_funding_slope_comparison.png'),
    )

    # ------------------------------------------------------------------
    # Save numerical results summary
    # ------------------------------------------------------------------
    summary_lines = [
        "=== EXP 1: Theoretical Replication Results ===",
        f"Residual identity error: {identity_result['identity_error']:.2e}",
        f"Symbolic identity holds: {sym_result['identity_holds']}",
        f"Weak fairness (martingale) passes: {wf_identity['weak_fairness_check']['passes']}",
        f"Weak fairness (jump) E[Delta_e] = {wf_jump['E_Delta_e']:.4f} (D={wf_jump['D']})",
        f"Strong fairness (jump) Delta_e = {sf_jump['Delta_e_tau']:.4f}",
        f"Counterexample deficit = {ce['deficit']:.4f}",
        f"Discrepancy (uniform density) = {disc:.4f} >= D={D_d}",
        f"Min discrepancy (Gaussian family) = {opt['min_discrepancy']:.4f}",
        f"Fair slope sign: {slope_sign_fair} (negative)",
        f"Deployed slope sign: {slope_sign_deploy} (positive)",
    ]
    out_path = RESULTS_DIR / 'exp1_results.txt'
    out_path.write_text('\n'.join(summary_lines))
    print(f"\nSummary saved to {out_path}")

    return results


if __name__ == '__main__':
    results = run_exp1()
    print('\nExp 1 complete. Results saved to results/')
