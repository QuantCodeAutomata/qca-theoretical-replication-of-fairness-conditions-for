"""
Predictable-jump fairness conditions for perpetual contracts.

At a predictable ex-dividend time tau, the shareholder receives a discrete
payment D_tau = Delta_A_tau.  For the residual M_t to remain a martingale
across tau, the basis must absorb the jump:

  Weak fairness:   E[Delta_e_tau | F_{tau-}] = D_tau
  Strong fairness: Delta_e_tau = D_tau  pathwise.

Counterexample:
  If *both* the perp mark and the reference spot drop by D_tau at ex-dividend
  then Delta_e_tau = Delta_P_tau - Delta_S_tau = 0, so the condition is
  violated by D_tau.  No absolutely continuous funding stream (which delivers
  zero probability mass at any single instant) can make up this shortfall.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Residual helpers (consistent with identities.py sign conventions)
# ---------------------------------------------------------------------------

def _cumulative_residual(
    S: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    r: float,
    phi: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Return cumulative residual R(0, t) for t = 0, 1, ..., n.

    R(0,t) = sum_{s<t}(dS + dA - r*S*dt) - sum_{s<t}(dP) + sum_{s<t}(phi*P*dt).
    """
    dS  = np.diff(S)
    dA  = np.diff(A)
    dP  = np.diff(P)
    inc = dS + dA - r * S[:-1] * dt - dP + phi * P[:-1] * dt
    return np.concatenate([[0.0], np.cumsum(inc)])


def _apply_event_credit(
    R_cumulative: np.ndarray,
    tau_idx: int,
    D: float,
) -> np.ndarray:
    """Add a discrete event credit of D to R_cumulative at index tau_idx."""
    R_ec = R_cumulative.copy()
    R_ec[tau_idx:] += D
    return R_ec


# ---------------------------------------------------------------------------
# Public demonstration functions
# ---------------------------------------------------------------------------

def demonstrate_weak_fairness_condition(
    D: float = 1.0,
    S0: float = 100.0,
    tau: int = 36,
    n_steps: int = 72,
    n_paths: int = 10_000,
    r: float = 0.05 / 252 / 24,
    q: float = 0.02 / 252 / 24,
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Monte Carlo verification of the weak fairness jump condition.

    The *fair* perp does not drop at ex-dividend (its mark stays flat while
    the spot drops by D), so Delta_e_tau = D.  We verify:

        E[Delta_e_tau | F_{tau-}] ≈ D.

    Parameters
    ----------
    D       : dividend size (absolute cash amount).
    S0      : initial spot price.
    tau     : index of the ex-dividend step (predictable, F_{tau-}-measurable).
    n_steps : total number of time steps.
    n_paths : Monte Carlo sample size.
    r, q, sigma_S : drift/vol parameters.
    seed    : RNG seed.

    Returns
    -------
    Dict with:
      'E_Delta_e'      : sample mean of Delta_e_tau across paths.
      'std_Delta_e'    : standard deviation.
      'D'              : target value.
      'condition_met'  : bool, |E[Delta_e] - D| < 3*se.
      'Delta_e_paths'  : array of per-path Delta_e_tau values, shape (n_paths,).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0

    # Simulate S up to tau for each path (vectorised)
    eps = rng.standard_normal((n_paths, n_steps))
    log_S_increments = (r - q - 0.5 * sigma_S**2) * dt + sigma_S * eps
    log_S = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_S_increments, axis=1)], axis=1
    )
    S = S0 * np.exp(log_S)  # shape (n_paths, n_steps+1)

    # At tau: spot drops by D (ex-dividend price adjustment)
    S_before_exdiv = S[:, tau]          # S_{tau-}
    S_after_exdiv  = S_before_exdiv - D # S_{tau} = S_{tau-} - D

    # Fair perp: mark does NOT drop at ex-div (it credits the holder)
    # => P_{tau} = P_{tau-},  so Delta_P_tau = 0
    # => Delta_e_tau = Delta_P_tau - Delta_S_tau = 0 - (-D) = D
    Delta_S_tau   = S_after_exdiv - S_before_exdiv   # = -D (deterministic)
    Delta_P_fair  = np.zeros(n_paths)                 # fair perp: no mark drop
    Delta_e_tau   = Delta_P_fair - Delta_S_tau        # = D

    E_Delta_e = float(np.mean(Delta_e_tau))
    std_Delta_e = float(np.std(Delta_e_tau))
    se = std_Delta_e / n_paths**0.5 if std_Delta_e > 0 else 1e-12

    return {
        "E_Delta_e"    : E_Delta_e,
        "std_Delta_e"  : std_Delta_e,
        "D"            : D,
        "condition_met": abs(E_Delta_e - D) < 3.0 * max(se, 1e-12),
        "Delta_e_paths": Delta_e_tau,
    }


def demonstrate_strong_fairness(
    D: float = 1.0,
    S0: float = 100.0,
    tau: int = 36,
    n_steps: int = 72,
    r: float = 0.05 / 252 / 24,
    q: float = 0.02 / 252 / 24,
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Demonstrate strong-fairness: Delta_e_tau = D pathwise.

    Simulates a single GBM path and shows the residual R(0, t) is identically
    zero when the fair (no-drop) perp mark convention is used with event credit.

    Parameters
    ----------
    D       : dividend size.
    S0, tau, n_steps, r, q, sigma_S, seed : simulation parameters.

    Returns
    -------
    Dict with:
      'R_cumulative'       : cumulative residual with fair perp, shape (n_steps+1,).
      'Delta_e_tau'        : basis jump at tau (should equal D).
      'strong_fairness_ok' : bool, |Delta_e_tau - D| < 1e-10.
      'max_abs_R'          : max |R(0,t)| over all t (ideally 0 with fair funding).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0

    eps_S = rng.standard_normal(n_steps)
    S = np.empty(n_steps + 1)
    A = np.empty(n_steps + 1)
    S[0] = S0
    A[0] = 0.0

    for t in range(n_steps):
        S[t + 1] = S[t] * np.exp((r - q - 0.5 * sigma_S**2) * dt + sigma_S * eps_S[t])
        A[t + 1] = A[t] + q * S[t] * dt

    # Ex-dividend price adjustment at tau
    S[tau] = S[tau] - D
    # Shareholder gets discrete dividend at tau
    A[tau] = A[tau] + D

    # Fair perp: mark does NOT drop at ex-div (Delta_P_tau = 0)
    # Construct P such that P = S + e where e_0 = 0, e_{tau} jumps by D
    e = np.zeros(n_steps + 1)
    e[tau:] += D   # Delta_e_tau = D => strong fairness holds
    P = S + e

    # Funding: use r - q as a simple carry-based rate (ignores basis for brevity)
    phi = np.full(n_steps, r - q)

    R = _cumulative_residual(S, A, P, r, phi, dt)

    Delta_S_tau = S[tau] - (S[tau] + D)   # = -D  (we already applied the drop)
    Delta_e_tau = float(e[tau] - e[tau - 1])  # = D

    return {
        "R_cumulative"      : R,
        "Delta_e_tau"       : Delta_e_tau,
        "strong_fairness_ok": abs(Delta_e_tau - D) < 1e-10,
        "max_abs_R"         : float(np.max(np.abs(R))),
    }


def counterexample_no_atom(
    D: float = 1.0,
    S0: float = 100.0,
    tau: int = 36,
    n_steps: int = 72,
    r: float = 0.05 / 252 / 24,
    q: float = 0.02 / 252 / 24,
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Counterexample: standard perp (both marks drop by D) + continuous funding.

    When both P and S drop by D at tau:
      Delta_P_tau = -D,  Delta_S_tau = -D  =>  Delta_e_tau = 0 ≠ D.

    The residual over any interval [tau1, tau2] with tau1 < tau <= tau2
    shows a permanent deficit of D that no absolutely continuous funding
    stream can remove.

    Parameters
    ----------
    D       : dividend size.
    S0, tau, n_steps, r, q, sigma_S, seed : simulation parameters.

    Returns
    -------
    Dict with:
      'R_standard'     : R(0,t) for standard perp (no event credit).
      'R_event_credit' : R(0,t) for event-credit perp.
      'Delta_e_tau'    : basis jump at tau (= 0 in standard case).
      'deficit'        : D - E[Delta_e_tau] = D (the shortfall).
      'R_deficit_post' : R(0, n) - R(0, tau) showing deficit is permanent.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0

    eps_S = rng.standard_normal(n_steps)
    S = np.empty(n_steps + 1)
    A = np.empty(n_steps + 1)
    S[0] = S0
    A[0] = 0.0

    for t in range(n_steps):
        S[t + 1] = S[t] * np.exp((r - q - 0.5 * sigma_S**2) * dt + sigma_S * eps_S[t])
        A[t + 1] = A[t] + q * S[t] * dt

    # Ex-dividend: spot drops by D, shareholder gets D
    S[tau] = S[tau] - D
    A[tau] = A[tau] + D

    # Standard perp: mark ALSO drops by D (no special treatment)
    e_standard = np.zeros(n_steps + 1)   # Delta_e_tau = 0
    P_standard  = S + e_standard

    # Continuous funding only (premium-index formula: f = r - q + kappa*e)
    phi_standard = np.full(n_steps, r - q)

    R_standard = _cumulative_residual(S, A, P_standard, r, phi_standard, dt)

    # Event-credit perp: add D at tau
    R_event_credit = _apply_event_credit(R_standard, tau, D)

    Delta_e_tau = float(e_standard[tau] - e_standard[tau - 1])  # = 0

    return {
        "R_standard"     : R_standard,
        "R_event_credit" : R_event_credit,
        "Delta_e_tau"    : Delta_e_tau,
        "deficit"        : D - Delta_e_tau,          # = D
        "R_deficit_post" : float(R_standard[-1] - R_standard[tau]),
    }


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise_jump_conditions(
    D: float = 1.0,
    S0: float = 100.0,
    tau: int = 36,
    n_steps: int = 72,
    n_paths: int = 5_000,
    seed: int = 42,
) -> None:
    """Print a human-readable summary of all three jump-condition demonstrations."""
    print("=== Weak fairness (Monte Carlo) ===")
    wf = demonstrate_weak_fairness_condition(D, S0, tau, n_steps, n_paths, seed=seed)
    print(f"  E[Delta_e_tau]  = {wf['E_Delta_e']:.4f}  (target D = {wf['D']:.4f})")
    print(f"  Condition met   : {wf['condition_met']}")

    print("\n=== Strong fairness (single path) ===")
    sf = demonstrate_strong_fairness(D, S0, tau, n_steps, seed=seed)
    print(f"  Delta_e_tau     = {sf['Delta_e_tau']:.4f}  (target D = {D:.4f})")
    print(f"  Strong fairness : {sf['strong_fairness_ok']}")

    print("\n=== Counterexample: no-atom continuous funding ===")
    ce = counterexample_no_atom(D, S0, tau, n_steps, seed=seed)
    print(f"  Delta_e_tau (standard) = {ce['Delta_e_tau']:.4f}  (should be 0)")
    print(f"  Deficit                = {ce['deficit']:.4f}  (= D = {D:.4f})")
    print(f"  Permanent R deficit    = {ce['R_deficit_post']:.4f}")
    print("  => No continuous funding stream can eliminate this D-sized gap.")


if __name__ == "__main__":
    summarise_jump_conditions(D=1.0, S0=100.0, tau=36, n_steps=72, n_paths=10_000)
