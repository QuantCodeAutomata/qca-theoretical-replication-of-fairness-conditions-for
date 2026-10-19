"""Exp 4: Empirical residual heatmaps."""
import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

from empirical.residuals import (
    compute_residual,
    build_residual_matrix,
    classify_straddling,
    compute_straddling_stats,
)
from empirical.counterfactual_atom import (
    insert_synthetic_atom,
    compute_counterfactual_residual_matrix,
    block_collapse_analysis,
)
from empirical.cross_venue_compare import generate_synthetic_example
from empirical.heatmaps import (
    plot_residual_heatmap,
    plot_before_after_heatmaps,
    plot_cross_venue_comparison,
)
from data.dividends import compute_expected_atom, compute_event_size_bps


def run_exp4():
    print('=== EXP 4: Empirical Residual Heatmaps ===')
    results = {}

    # ------------------------------------------------------------------
    # 1. Generate synthetic event window (±24 h around ex-date)
    # ------------------------------------------------------------------
    print('\n1. Generating synthetic residual matrices...')
    D_val, S0_val = 5.0, 230.0
    synth = generate_synthetic_example(
        D=D_val,
        S0=S0_val,
        n_steps=48,
        tau=24,
        sigma_b=0.0006,
        phi_ar1=0.72,
        r=0.0,
    )
    print(f"   Dividend D = {synth['D']:.2f}")
    print(f"   Fair atom phi_D = {synth['phi_atom']:.6f}")
    print(f"   Event index tau = {synth['event_idx']}")

    R_no_atom   = synth['R_matrix_no_atom']
    R_with_atom = synth['R_matrix_with_atom']
    is_strad    = synth['is_straddling']
    timestamps  = synth['timestamps']
    tau_idx     = synth['event_idx']

    # ------------------------------------------------------------------
    # 2. Straddling statistics before atom
    # ------------------------------------------------------------------
    print('\n2. Straddling statistics (before atom insertion)...')
    stats_before = synth['stats_no_atom']
    stats_after  = synth['stats_with_atom']
    print(f"   Before: straddling_mean = {stats_before['straddling_mean']:.6f}  "
          f"non-strad_mean = {stats_before['non_straddling_mean']:.6f}")
    print(f"   After : straddling_mean = {stats_after['straddling_mean']:.6f}  "
          f"non-strad_mean = {stats_after['non_straddling_mean']:.6f}")

    collapse = synth['collapse']
    print(f"\n3. Block collapse after atom insertion:")
    print(f"   Reduction (before - after mean): {collapse['reduction']:.6f}")
    if not np.isnan(collapse['reduction_pct']):
        print(f"   Reduction %:                     {collapse['reduction_pct']:.1f}%")
    print(f"   After straddling mean closer to zero: "
          f"{abs(stats_after['straddling_mean']) < abs(stats_before['straddling_mean'])}")

    results['block_collapse'] = {
        'before_strad_mean': stats_before['straddling_mean'],
        'after_strad_mean': stats_after['straddling_mean'],
        'reduction': collapse['reduction'],
        'reduction_pct': collapse['reduction_pct'],
    }

    # ------------------------------------------------------------------
    # 4. Residual decomposition on a sample interval
    # ------------------------------------------------------------------
    print('\n4. Sample residual decomposition...')
    P_arr = synth['P']
    S_arr = synth['S']
    fund  = synth['funding_no_atom']

    # Straddling interval: entry before tau, exit after tau
    t1, t2 = tau_idx - 5, tau_idx + 5
    decomp = compute_residual(
        t1_idx=t1,
        t2_idx=t2,
        timestamps=timestamps,
        P=P_arr,
        S=S_arr,
        funding_rates=fund,
        r=0.0,
        A_jumps=None,
    )
    print(f"   Interval [{t1}, {t2}] (straddles tau={tau_idx}):")
    print(f"   R_total        = {decomp['R_total']:.6f}")
    print(f"   dividend_leg   = {decomp['dividend_leg']:.6f}")
    print(f"   basis_change   = {decomp['basis_change_leg']:.6f}")
    print(f"   funding_leg    = {decomp['funding_leg']:.6f}")
    print(f"   financing_leg  = {decomp['financing_leg']:.6f}")

    # Verify decomposition
    R_check = (decomp['dividend_leg'] - decomp['basis_change_leg']
               + decomp['funding_leg'] - decomp['financing_leg'])
    print(f"   Decomposition error = {abs(decomp['R_total'] - R_check):.2e}")
    results['residual_decomp'] = decomp

    # ------------------------------------------------------------------
    # 5. Save heatmap figures
    # ------------------------------------------------------------------
    print('\n5. Saving residual heatmap figures...')

    plot_residual_heatmap(
        R_matrix=R_no_atom,
        is_straddling=is_strad,
        timestamps=timestamps,
        event_idx=tau_idx,
        title='Residual Heatmap (No Atom)',
        save_path=str(RESULTS_DIR / 'heatmap_no_atom.png'),
    )

    plot_residual_heatmap(
        R_matrix=R_with_atom,
        is_straddling=is_strad,
        timestamps=timestamps,
        event_idx=tau_idx,
        title='Residual Heatmap (With Atom)',
        save_path=str(RESULTS_DIR / 'heatmap_with_atom.png'),
    )

    plot_before_after_heatmaps(
        R_before=R_no_atom,
        R_after=R_with_atom,
        is_straddling=is_strad,
        timestamps=timestamps,
        event_idx=tau_idx,
        title_prefix='Synthetic AAPL',
        save_path=str(RESULTS_DIR / 'heatmap_before_after.png'),
    )

    # Cross-venue comparison bar chart
    venue_results = {
        'Standard (no atom)': {
            'atom_present': False,
            'phi_observed': 0.0,
            'stats': stats_before,
            'counterfactual_stats': collapse,
        },
        'Event-credit (with atom)': {
            'atom_present': True,
            'phi_observed': synth['phi_atom'],
            'stats': stats_after,
            'counterfactual_stats': collapse,
        },
    }
    plot_cross_venue_comparison(
        results_by_venue=venue_results,
        save_path=str(RESULTS_DIR / 'cross_venue_comparison.png'),
    )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    out_path = RESULTS_DIR / 'exp4_results.txt'
    out_path.write_text(
        f"=== EXP 4: Residual Heatmap Results ===\n"
        f"D = {D_val}, tau = {tau_idx}\n"
        f"Straddling mean before: {stats_before['straddling_mean']:.6f}\n"
        f"Straddling mean after:  {stats_after['straddling_mean']:.6f}\n"
        f"Reduction: {collapse['reduction']:.6f}\n"
    )
    print(f"\nSummary saved to {out_path}")

    return results


if __name__ == '__main__':
    run_exp4()
    print('\nExp 4 complete. Results saved to results/')
