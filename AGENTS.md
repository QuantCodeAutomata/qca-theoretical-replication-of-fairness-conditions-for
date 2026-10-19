# Repository Knowledge — Equity Perpetual Fairness Conditions

## Project Overview
Research codebase studying equity perpetual funding fairness conditions,
specifically discrete dividend atoms on crypto exchanges (Binance, Hyperliquid, OKX).

## Module Map

```
data/
  dividends.py   — DIVIDEND_CALENDAR (5 equities), compute_expected_atom(), compute_event_size_bps()
  download.py    — REST API fetchers for Binance fapi + Hyperliquid; all return empty DataFrame on error

empirical/
  atom_detection.py     — Exp 3: Table 1 replication; synthetic fallback when API unavailable
  residuals.py          — Exp 4: compute_residual(), build_residual_matrix(), classify_straddling()
  counterfactual_atom.py — Exp 4: insert_synthetic_atom(), block_collapse_analysis()
  heatmaps.py           — matplotlib/seaborn heatmaps saved to /workspace/project/results/
  cross_venue_compare.py — compare_venues(), generate_synthetic_example() (fully self-contained)

panel/
  universe.py         — HYPERLIQUID_EQUITY_PERPS (13 tickers), build_panel_universe()
  premium_series.py   — compute_basis(), compute_relative_basis(), build_premium_series()
  ar1_estimation.py   — estimate_ar1() via statsmodels OLS, fit_panel_ar1()
  funding_rule_docs.py — VENUE_FUNDING_RULES dict, get_funding_rule(), generate_symbol_summary_table()
```

## Key Design Decisions

### Data sourcing
- **No Massive API** — all exchange calls use direct REST via `requests` library
- Binance USDT-M: `https://fapi.binance.com/fapi/v1/{fundingRate,markPriceKlines,indexPriceKlines}`
- Hyperliquid: `POST https://api.hyperliquid.xyz/info` with JSON body type field
- All fetchers return **empty DataFrame with documented schema** on any error (never raise)
- Timeout: 30 s on all requests

### Core formula
```
phi_atom = -D / (Mark - D)        # fair dividend atom
R(t1,t2) = ΔA - Δe + Σphi - Σr  # no-arb residual (eq. 2)
```

### Straddling convention
`classify_straddling(t1, t2, event)` returns True iff `t1 < event <= t2`
(position entered strictly before event, exited at or after event).

### Residual matrix layout
- `R[i,j]` filled only for `j > i` (strict upper triangle)
- Lower triangle + diagonal = NaN

### Synthetic fallback (atom_detection.py)
When exchange API returns no data, mark price is back-calculated:
`mark = D - D/phi_approx`  (from paper's expected_atom_approx field)
Results flagged `data_source='synthetic_paper'`.

### Heatmap output
All figures saved to `/workspace/project/results/` with Agg backend (headless safe).

## Dependencies
- Python: requests, numpy, pandas, matplotlib, seaborn, statsmodels
- All confirmed installed (2025-Q2 versions)

## Running path
```bash
cd /workspace/project
python -c "import sys; sys.path.insert(0,'.')"  # required for intra-package imports
```

## Test patterns
```python
# Zero residual: flat P and S, zero funding → R=0 for all intervals
# AR(1) recovery: simulate with phi=0.85, recover phi in [0.7,1.0]
# Atom insertion: modified[event_idx] = original + phi_atom, rest unchanged
```

---

## Theory & Simulation Modules (added 2025)

### Module Map
```
theory/
  identities.py    – residual identity R(τ₁,τ₂), process M_t, weak/strong fairness
  jumps.py         – predictable-dividend jump conditions, counterexample
  discrepancy.py   – interval-discrepancy Δ(μ,ν), infimum analysis
  ou_fair_rule.py  – OU/AR(1) fair funding f_fair(b), kernel projection benchmark
simulation/
  dividend_paths.py  – three P&L paths (shareholder, standard perp, event-credit)
  basis_ou.py        – mean-reverting basis OU simulation + AR(1) estimation
  funding_rules.py   – deployed vs fair funding, slope comparison
  figures.py         – four research figures → results/
results/             – PNG outputs
```

### Key Design Decisions

**theory/identities.py**
- Sign convention: long **pays** funding φ_t·P_t·dt to short; basis e_t = P_t − S_t.
- `symbolic_verify_residual_identity()` returns dict; keys `identity_holds` and `R_equals_M_diff` are bools.
- Numerical identity error ≈ 6.8e-15 (machine precision); threshold 1e-10.

**theory/jumps.py**
- `demonstrate_weak_fairness_condition(D, S0, tau, n_steps, n_paths)` — Monte Carlo E[Δe_τ]=D.
- `counterexample_no_atom` — shows deficit = D when both perp mark and index drop by D.

**theory/discrepancy.py**
- `compute_discrepancy` (grid-based) shows ~0.993 for uniform density (grid resolution artifact).
- `compute_discrepancy_fast` correctly converges to 1.0 = D as ε → 0.
- `verify_infimum_equals_D` uses tolerance 1e-4; `lower_bound_holds` is True.

**theory/ou_fair_rule.py**
- `symbolic_derive_fair_funding()['derivation_equiv']` is a Python **bool** (not sympy Eq).
- `projection_benchmark(Y_samples, P_samples, e_samples, x_grid)` — Nadaraya-Watson kernel with P² weights (spec-required signature).
- `projection_nonneg(Y_t)` — simple max(Y_t, 0) clipping (separate helper).
- Equation (8): f_fair(b) = (r−q) − (λb − ρσ_S σ_b)/(1+b); slope always negative when λ > 0, ρ = 0.

**simulation/dividend_paths.py**
- Shareholder wealth **must** subtract `cum_rS = cumsum(r·S·dt)` (financing cost).  Without this, deficit = D + sum(r·S·dt) ≠ D.
- `verify_dividend_loss` default tolerance = **1e-5** (FP accumulation ~3e-6 over 72 steps).
- Pass `tau` explicitly; auto-inference via gap column is fallback only.

**simulation/basis_ou.py**
- Half-life 2.1 h → λ = ln(2)/2.1 ≈ 0.330; φ_AR1 = exp(−λ) ≈ 0.72.
- `estimate_ar1_persistence(b_series)` uses statsmodels ARIMA(1,0,0).
- Need n_steps ≥ 500 to reliably estimate φ within ±0.15.

**simulation/funding_rules.py**
- `cumulative_funding(funding_rates, P, dt)` — `funding_rates` and `P` must have the **same** length N; returns N+1 array.
- Deployed slope κ > 0; fair slope = −λ/(1+b)² < 0: **opposite signs everywhere**.

### Running Modules
```bash
python -m theory.identities
python -m theory.jumps
python -m theory.discrepancy
python -m theory.ou_fair_rule
python -m simulation.dividend_paths
python -m simulation.basis_ou
python -m simulation.funding_rules
python -m simulation.figures       # generates all four PNGs to results/
```

### Common Pitfalls
1. `projection_benchmark` ≠ `projection_nonneg` — first is kernel regression (Nadaraya-Watson), second is simple `max(Y, 0)`. Tests of max(Y,0) must use `projection_nonneg`.
2. `derivation_equiv` is a Python `bool`, not `sympy.Eq`; check with `== True`.
3. Shareholder wealth requires `−cum_rS`; omitting it breaks the D-deficit identity.
4. `verify_dividend_loss` tolerance is 1e-5; 1e-6 fails due to float accumulation.
5. **D recovery in tests** — never compute `D = D_frac * df.loc[df["t"]==tau-1, "S"]`; S[tau-1] ≠ S[tau] pre-drop (differ by one GBM step). Recover D correctly with: `D = df.loc[df["t"]==tau,"event_credit_wealth"] - df.loc[df["t"]==tau,"standard_perp_wealth"]`.
6. **Weak fairness martingale test** — single-path bootstrap of sub-interval increments is NOT a reliable test; any single realization can drift orders of magnitude above the 3-SE threshold. Test by averaging `M_T − M_0` over ≥200 independent paths; the cross-path mean should be within 3 SE of zero.
7. **`compute_discrepancy` (grid-based)** gives ≈0.993 for D=1 (grid truncation error). Use `compute_discrepancy_fast` which adds shrinking-interval near-tau sampling → correctly converges to D. The fast version returns a dict; use `result["discrepancy"]`.
8. **`minimize_discrepancy` optimizer tolerance** — Nelder-Mead + grid discretization can undershoot `lower_bound` by up to 2e-4; the test assertion `gap >= -1e-6` is too tight. Use `OPT_TOL = 2e-4`.

### Test-Suite Status (all 37 pass)
```
tests/test_theory.py     (8 tests)   — covers residual identity, fairness, discrepancy, OU rule
tests/test_simulation.py (8 tests)   — dividend paths, OU basis, funding rules, cumulative
tests/test_empirical.py  (7 tests)   — atom detection, residual matrix, block collapse
tests/test_panel.py      (6+ tests)  — AR(1) panel, halflife, universe, summary table
```
