# Theoretical Replication of Fairness Conditions for Equity Perpetuals

A research repository replicating and extending the theoretical fairness conditions
for equity perpetual futures contracts, covering mathematical identities, simulation
experiments, empirical atom verification, and panel-level basis dynamics.

---

## Overview

Equity perpetual futures (equity perps) are crypto-native derivatives that track
equity spot prices.  Unlike traditional futures, perpetuals have no expiry and rely
on a _funding rate_ mechanism to keep the mark price anchored to the reference index.
When the underlying equity pays a discrete cash dividend, the reference index drops
by the dividend amount, creating a predictable discontinuity that no absolutely
continuous funding stream can fully absorb.

This project replicates five experiments from the theoretical paper:

| Exp | Title | Key question |
|-----|-------|-------------|
| 1 | Theoretical replication | Does the residual accounting identity hold?  What is the fair funding rule under OU basis dynamics? |
| 2 | Simulation replication | How large is the dividend-loss shortfall for a standard perp holder?  Does the OU fair rule have a negative slope? |
| 3 | Exchange-native atom verification | Does Binance post a discrete atom φ_D = −D/(Mark−D) at ex-dividend dates?  Do standard venues (Hyperliquid, OKX) match? |
| 4 | Empirical residual heatmaps | Do straddling residuals collapse to zero once the counterfactual fair atom is inserted? |
| 5 | Descriptive panel | What are the AR(1) half-life and mean-reversion speed of the relative basis across the 13-ticker Hyperliquid universe? |

---

## Directory Structure

```
project/
├── theory/                  # Mathematical theory modules
│   ├── identities.py        # Residual accounting identity (Eq. 2) and fair funding target (Eq. 4)
│   ├── jumps.py             # Predictable-jump fairness conditions (weak and strong)
│   ├── discrepancy.py       # Interval-discrepancy metric Δ(μ, ν) ≥ D
│   └── ou_fair_rule.py      # OU-based fair funding rule (Eq. 8) and projection benchmark
│
├── simulation/              # Monte Carlo and analytic simulation modules
│   ├── dividend_paths.py    # Dividend path comparison: standard vs event-credit perp
│   ├── basis_ou.py          # AR(1) / OU basis simulation and persistence estimation
│   ├── funding_rules.py     # Deployed vs fair funding rule implementations
│   └── figures.py           # Simulation figure generators
│
├── data/                    # Data ingestion and reference tables
│   ├── dividends.py         # Synthetic dividend calendar and atom-size helpers
│   └── download.py          # Exchange REST API clients (Binance, Hyperliquid)
│
├── empirical/               # Empirical analysis modules
│   ├── atom_detection.py    # Table 1 replication: dividend atom detection by venue
│   ├── residuals.py         # Discrete residual computation and straddling classification
│   ├── counterfactual_atom.py  # Synthetic atom insertion and block-collapse analysis
│   ├── heatmaps.py          # Residual heatmap figure generators
│   └── cross_venue_compare.py  # Cross-venue summary and synthetic example
│
├── panel/                   # Panel-level basis and AR(1) analysis
│   ├── universe.py          # Equity perp universe discovery (13-ticker HL panel)
│   ├── premium_series.py    # Absolute and relative basis series construction
│   ├── ar1_estimation.py    # AR(1) estimation, half-life, and λ computation
│   └── funding_rule_docs.py # Venue funding rule documentation and slope classification
│
├── exp/                     # Experiment runner scripts
│   ├── exp1_theoretical.py  # Exp 1: theoretical replication
│   ├── exp2_simulation.py   # Exp 2: simulation replication
│   ├── exp3_atom_verification.py  # Exp 3: atom verification
│   ├── exp4_heatmaps.py     # Exp 4: empirical residual heatmaps
│   └── exp5_descriptive_panel.py  # Exp 5: descriptive panel
│
├── tests/                   # Pytest test suite
│   ├── test_theory.py       # Unit tests for theory modules
│   ├── test_simulation.py   # Unit tests for simulation modules
│   ├── test_empirical.py    # Unit tests for empirical modules
│   └── test_panel.py        # Unit tests for panel modules
│
├── results/                 # Output directory (figures, tables — git-ignored)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd project

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run Experiments

Each experiment is a self-contained script that writes figures and tables to
`/workspace/project/results/`.

```bash
# Exp 1 — Theoretical replication of fairness conditions
python exp/exp1_theoretical.py

# Exp 2 — Simulation replication of long-perp dividend loss
python exp/exp2_simulation.py

# Exp 3 — Exchange-native event-atom verification (Table 1)
python exp/exp3_atom_verification.py

# Exp 4 — Empirical residual heatmaps
python exp/exp4_heatmaps.py

# Exp 5 — Descriptive panel of premium mean reversion
python exp/exp5_descriptive_panel.py
```

Results are written to `results/`.  Exchange API calls in Exp 3–5 fall back to
synthetic paper-stated values when live data is unavailable.

---

## How to Run Tests

```bash
# Run the full test suite from the project root
pytest tests/ -v

# Run a single test module
pytest tests/test_theory.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

All tests are self-contained and do not require live exchange connectivity.

---

## Results

After running all experiments the `results/` directory contains:

| File | Description |
|------|-------------|
| `residual_identity_verification.txt` | Algebraic and numerical identity checks |
| `discrepancy_illustration.png` | Shrinking-interval lower-bound plot (Δ ≥ D) |
| `funding_slope_comparison.png` | Fair vs deployed funding slope comparison |
| `dividend_path_comparison.png` | Standard vs event-credit perp wealth paths |
| `funding_rule_comparison.png` | f_fair(b) vs f_deploy(b) over basis range |
| `table1_atom_verification.csv` | Table 1: atom presence by venue and ex-date |
| `heatmap_no_atom.png` | Residual heatmap before counterfactual insertion |
| `heatmap_with_atom.png` | Residual heatmap after counterfactual insertion |
| `panel_ar1_summary.csv` | AR(1) estimates for 13-ticker HL panel |
| `panel_summary_figure.png` | Half-life and persistence bar charts |

---

## Key Mathematical Results

### Residual Accounting Identity (Eq. 2)
For any interval [τ₁, τ₂]:

    R(τ₁, τ₂) = M_{τ₂} − M_{τ₁}

where M_t = A_t − e_t + ∫₀ᵗ (φ_u P_u − r_u S_u) du.

### Fair Funding Target (Eq. 4)
The fair funding rate Y_t that drives E[R] = 0:

    Y_t = (r_t S_t + μ^e_t − a_t) / P_t

### Interval-Discrepancy Lower Bound
For any absolutely continuous measure μ competing with ν = D · δ_τ:

    Δ(μ, ν) ≥ D

The infimum is not attained; no continuous funding stream can deliver the atom.

### OU Fair Funding Rule (Eq. 8)
Under AR(1) basis dynamics b_t with mean-reversion speed λ:

    f_fair(b_t) = (r − q − λ b_t) / (1 + b_t)
    f_fair(b_t) ≈ (r − q) − λ b_t   [small-b approximation]

The slope ∂f_fair/∂b < 0 for λ > 0, opposite to all deployed venue rules.

### Deployed Venue Rule
All three covered venues (Hyperliquid, Binance, OKX) use positive-slope rules:

    f_deploy(b) ∝ clamp(b, lower, upper) + interest_rate

---

## License

Research use only.  See LICENSE for details.
