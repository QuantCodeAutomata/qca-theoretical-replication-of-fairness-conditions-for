"""Exp 3: Exchange-native event-atom verification."""
import sys
sys.path.insert(0, '/workspace/project')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)

from data.dividends import get_dividend_calendar, compute_expected_atom, compute_event_size_bps
from empirical.atom_detection import (
    build_event_master_table,
    detect_binance_dividend_atom,
    check_standard_venue_atom,
    generate_table1_replication,
    validate_atom_match,
)


def run_exp3():
    print('=== EXP 3: Exchange-Native Atom Verification ===')

    # ------------------------------------------------------------------
    # 1. Load dividend calendar
    # ------------------------------------------------------------------
    print('\n1. Loading dividend calendar...')
    div_cal = get_dividend_calendar()
    print(div_cal.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Expected atom computation
    # ------------------------------------------------------------------
    print('\n2. Computing expected funding atoms...')
    for _, row in div_cal.iterrows():
        D    = row['D']
        phi  = row['expected_atom_approx']
        # Back-calculate approximate mark price from the paper's atom
        mark_approx = D - D / phi  # phi = -D/(Mark-D) => Mark = D - D/phi
        bps  = compute_event_size_bps(D, mark_approx)
        print(f"   {row['symbol']:6s} D={D:.3f}  phi_expected={phi:.5f}  "
              f"size={bps:.1f} bps")

    # ------------------------------------------------------------------
    # 3. Build event master table
    # ------------------------------------------------------------------
    print('\n3. Building event master table...')
    master = build_event_master_table(div_cal)
    print(f"   {len(master)} venue/event rows")

    # ------------------------------------------------------------------
    # 4. Table 1 replication: atom presence by venue
    # ------------------------------------------------------------------
    print('\n4. Generating Table 1 replication...')
    table1 = generate_table1_replication(div_cal, venues=['binance', 'hyperliquid'])

    print('\n   === TABLE 1: Atom Verification ===')
    print(table1.to_string(index=False))

    # Validate atom matching
    print('\n5. Atom match validation (tolerance = 5e-6)...')
    for _, row in table1.iterrows():
        match = validate_atom_match(
            expected=row['expected_atom'],
            observed=row['observed_atom'],
            tolerance=5e-6,
        )
        status = 'MATCH' if match else 'MISS'
        print(f"   [{status}] {row['venue']:12s} {row['underlying']:6s}  "
              f"expected={row['expected_atom']:.5f}  "
              f"observed={row['observed_atom']:.5f}  "
              f"|error|={row['absolute_error']:.2e}  "
              f"source={row['data_source']}")

    # ------------------------------------------------------------------
    # 6. Venue-level summary
    # ------------------------------------------------------------------
    print('\n6. Venue-level atom presence summary:')
    venue_summary = (
        table1.groupby('venue')['atom_present']
        .agg(['sum', 'count'])
        .rename(columns={'sum': 'atom_present_count', 'count': 'total_events'})
    )
    venue_summary['atom_rate'] = (
        venue_summary['atom_present_count'] / venue_summary['total_events']
    )
    print(venue_summary.to_string())

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_csv = RESULTS_DIR / 'table1_atom_verification.csv'
    table1.to_csv(out_csv, index=False)
    print(f'\nTable 1 saved to {out_csv}')

    out_summary = RESULTS_DIR / 'exp3_results.txt'
    out_summary.write_text(
        f"=== EXP 3: Atom Verification Results ===\n"
        f"{venue_summary.to_string()}\n\n"
        f"Full table:\n{table1.to_string(index=False)}\n"
    )
    print(f'Summary saved to {out_summary}')

    return table1


if __name__ == '__main__':
    run_exp3()
    print('\nExp 3 complete. Results saved to results/')
