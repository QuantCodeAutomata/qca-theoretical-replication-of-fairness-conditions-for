"""
Interval-discrepancy metric for comparing a signed measure mu to the Dirac mass nu.

Definition (paper notation):
    Delta(mu, nu) = sup_{a < b} |mu((a, b]) - nu((a, b])|

Event-credit measure:  nu = D * delta_tau   (Dirac mass of size D at tau).

Key analytical result:
    For any absolutely continuous measure mu,  Delta(mu, nu) >= D.
    The infimum  inf_{mu << Leb} Delta(mu, nu) = D  is not attained.

Intuition: shrink the interval (a, b] down to (tau-eps, tau].
  mu((tau-eps, tau]) -> 0  (absolutely continuous, no point mass possible).
  nu((tau-eps, tau]) = D   (the Dirac mass always lies in the interval).
  => |mu - nu| -> D  as eps -> 0.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from typing import Callable, Dict, Tuple


# ---------------------------------------------------------------------------
# Core discrepancy computation
# ---------------------------------------------------------------------------

def compute_discrepancy(
    mu_density: Callable[[float], float],
    tau: float,
    D: float,
    grid: np.ndarray,
) -> float:
    """
    Compute the interval-discrepancy Delta(mu, nu) where nu = D * delta_tau.

    Evaluates  sup_{a < b in grid} |mu((a,b]) - nu((a,b])|  numerically.

    Parameters
    ----------
    mu_density : Lebesgue density m(t) of the absolutely continuous measure mu.
                 Must be non-negative and integrable over the grid range.
    tau        : location of the Dirac mass in nu.
    D          : mass of the Dirac atom (D > 0).
    grid       : sorted 1-D array of evaluation points spanning [0, horizon].

    Returns
    -------
    Estimated supremum of |mu((a,b]) - nu((a,b])| over all grid pairs (a,b).
    """
    grid = np.asarray(grid, dtype=float)
    n = len(grid)
    max_disc = 0.0

    # Pre-compute cumulative mu on the grid via trapezoidal quadrature.
    # mu_cdf[i] = mu((grid[0], grid[i]])
    density_vals = np.vectorize(mu_density)(grid)
    mu_cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density_vals[:-1] + density_vals[1:]) * np.diff(grid)
    )])

    for i in range(n):
        for j in range(i + 1, n):
            a, b = grid[i], grid[j]
            mu_ab = mu_cdf[j] - mu_cdf[i]
            nu_ab = D if a < tau <= b else 0.0
            disc = abs(mu_ab - nu_ab)
            if disc > max_disc:
                max_disc = disc

    return max_disc


def compute_discrepancy_fast(
    mu_density: Callable[[float], float],
    tau: float,
    D: float,
    grid: np.ndarray,
) -> Dict[str, float]:
    """
    Faster discrepancy computation that exploits the structure of nu = D*delta_tau.

    The supremum decomposes into three cases:
      (A) b < tau  : nu = 0,  disc = mu((a,b])  (maximised by taking a=0, b->tau-).
      (B) a >= tau : nu = 0,  disc = mu((a,b])  (maximised by a->tau+, b=T).
      (C) a < tau <= b: nu = D, disc = |mu((a,b]) - D|.
              - As a -> tau-, b fixed: mu((a,b]) -> mu((tau,b]) => |mu-D| -> |mu(tau,b]-D|.
              - As b -> tau+, a fixed: mu((a,b]) -> mu((a,tau]) => |mu-D| -> |mu(a,tau]-D|.
              - Shrinking both to tau: mu -> 0, disc -> D.

    Returns
    -------
    Dict with 'discrepancy', 'max_left_case', 'max_right_case', 'max_straddling'.
    """
    grid = np.asarray(grid, dtype=float)
    density_vals = np.vectorize(mu_density)(grid)
    mu_cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density_vals[:-1] + density_vals[1:]) * np.diff(grid)
    )])

    tau_idx = np.searchsorted(grid, tau, side="right")  # first index > tau

    # Case A: both endpoints < tau
    mu_left_total  = mu_cdf[tau_idx] - mu_cdf[0]
    max_left_case  = float(mu_left_total)  # maximised by a=0, b just before tau

    # Case B: both endpoints >= tau
    mu_right_total = mu_cdf[-1] - mu_cdf[tau_idx]
    max_right_case = float(mu_right_total)

    # Case C: straddling tau — scan all (a<tau, b>=tau) pairs on the grid
    max_strad = 0.0
    for i in range(tau_idx + 1):          # a <= tau
        for j in range(tau_idx, len(grid)):  # b >= tau
            if i >= j:
                continue
            mu_ab = mu_cdf[j] - mu_cdf[i]
            disc  = abs(mu_ab - D)
            if disc > max_strad:
                max_strad = disc

    # Near-tau straddling: sample a sequence of shrinking intervals (tau-eps, tau].
    # As eps -> 0, mu -> 0 (absolutely continuous) but nu = D, so disc -> D.
    # This reveals the true supremum that the coarse grid misses.
    a_lo = grid[0]
    for log_eps in np.linspace(-8, np.log10(tau - a_lo + 1e-15), 25):
        eps = 10.0 ** log_eps
        left_a = max(a_lo, tau - eps)
        mu_left, _ = integrate.quad(mu_density, left_a, tau)
        disc_left = abs(mu_left - D)   # interval (tau-eps, tau], nu = D
        max_strad = max(max_strad, disc_left)

    overall = max(max_left_case, max_right_case, max_strad)
    return {
        "discrepancy"    : overall,
        "max_left_case"  : max_left_case,
        "max_right_case" : max_right_case,
        "max_straddling" : max_strad,
    }


# ---------------------------------------------------------------------------
# Analytical lower bound proof
# ---------------------------------------------------------------------------

def analytical_lower_bound(
    mu_density: Callable[[float], float],
    tau: float,
    D: float,
    eps_values: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Demonstrate Delta(mu, nu) >= D analytically via shrinking intervals.

    For eps > 0, compute mu((tau-eps, tau]) and |mu((tau-eps, tau]) - D|.
    As eps -> 0 the latter -> D, proving the lower bound.

    Parameters
    ----------
    mu_density : density of mu (absolutely continuous).
    tau        : Dirac atom location.
    D          : Dirac mass.
    eps_values : shrinking half-widths; defaults to logspace(-4, 0, 40).

    Returns
    -------
    Dict with 'eps', 'mu_interval', 'discrepancy_contribution', 'lower_bound'.
    """
    if eps_values is None:
        eps_values = np.logspace(-4, 0, 40)

    mu_interval = np.array([
        integrate.quad(mu_density, max(0.0, tau - eps), tau)[0]
        for eps in eps_values
    ])
    disc_contrib = np.abs(mu_interval - D)

    return {
        "eps"                     : eps_values,
        "mu_interval"             : mu_interval,
        "discrepancy_contribution": disc_contrib,
        "lower_bound"             : D,   # limiting value
    }


# ---------------------------------------------------------------------------
# Optimal AC approximation search
# ---------------------------------------------------------------------------

def _gaussian_density(t: float, mu_loc: float, sigma_loc: float, D: float) -> float:
    """Gaussian density normalised to total mass D on the real line."""
    return (D / (sigma_loc * (2.0 * np.pi) ** 0.5)) * np.exp(
        -0.5 * ((t - mu_loc) / sigma_loc) ** 2
    )


def minimize_discrepancy(
    tau: float,
    D: float,
    horizon: float,
    n_grid: int = 200,
) -> Dict[str, object]:
    """
    Find the absolutely continuous measure that minimises Delta(mu, nu).

    Parameterises mu as a Gaussian density N(mu_loc, sigma_loc^2) with total
    mass D on [0, horizon], then minimises the discrepancy over (mu_loc, sigma_loc).

    The theoretical minimum is D (not achieved by any AC measure).

    Parameters
    ----------
    tau     : Dirac atom location.
    D       : Dirac mass (D > 0).
    horizon : right endpoint of the time domain.
    n_grid  : grid resolution for discrepancy evaluation.

    Returns
    -------
    Dict with:
      'optimal_loc'   : best Gaussian centre mu_loc.
      'optimal_sigma' : best Gaussian width sigma_loc.
      'min_discrepancy': discrepancy at optimum.
      'lower_bound'   : D (analytical lower bound).
      'gap'           : min_discrepancy - D (always >= 0).
      'candidates'    : array of (mu_loc, sigma_loc, discrepancy) tried.
    """
    grid = np.linspace(0.0, horizon, n_grid)

    def objective(params: np.ndarray) -> float:
        mu_loc, log_sigma = params
        sigma_loc = np.exp(log_sigma)
        density = lambda t: _gaussian_density(t, mu_loc, sigma_loc, D)
        res = compute_discrepancy_fast(density, tau, D, grid)
        return res["discrepancy"]

    # Candidate starting points: Gaussian centred just to the right of tau
    candidates = []
    for offset in [0.01, 0.1, 0.5, 1.0, -0.01, -0.1]:
        for log_sig in [-3.0, -1.5, -0.5, 0.5]:
            x0 = np.array([tau + offset * horizon, log_sig])
            try:
                res = optimize.minimize(
                    objective, x0,
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300},
                )
                candidates.append((res.x[0], np.exp(res.x[1]), res.fun))
            except Exception:
                pass

    candidates_arr = np.array(candidates)
    best_idx       = int(np.argmin(candidates_arr[:, 2]))
    best_loc       = float(candidates_arr[best_idx, 0])
    best_sigma     = float(candidates_arr[best_idx, 1])
    min_disc       = float(candidates_arr[best_idx, 2])

    return {
        "optimal_loc"    : best_loc,
        "optimal_sigma"  : best_sigma,
        "min_discrepancy": min_disc,
        "lower_bound"    : D,
        "gap"            : min_disc - D,
        "candidates"     : candidates_arr,
    }


# ---------------------------------------------------------------------------
# Numerical verification: best AC approximation >= D
# ---------------------------------------------------------------------------

def verify_infimum_equals_D(
    tau: float = 0.5,
    D: float = 1.0,
    horizon: float = 1.0,
    n_grid: int = 300,
) -> Dict[str, object]:
    """
    Verify inf_{mu << Leb} Delta(mu, nu) = D by showing:
      (1) Any AC measure has discrepancy >= D (lower bound via shrinking intervals).
      (2) Gaussian approximations concentrated near tau from the right approach D.

    Parameters
    ----------
    tau     : Dirac atom location in (0, horizon).
    D       : Dirac mass.
    horizon : time-domain right endpoint.
    n_grid  : grid resolution.

    Returns
    -------
    Dict with 'lower_bound_holds', 'limiting_discrepancy', 'infimum_approached'.
    """
    grid = np.linspace(0.0, horizon, n_grid)

    # Sequence of Gaussians concentrating at tau from the right
    sigmas      = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    discs_right = []
    discs_left  = []

    for sigma in sigmas:
        # Concentrated to the right of tau
        mu_loc_r = tau + sigma
        density_r = lambda t, m=mu_loc_r, s=sigma: _gaussian_density(t, m, s, D)
        res_r = compute_discrepancy_fast(density_r, tau, D, grid)
        discs_right.append(res_r["discrepancy"])

        # Concentrated to the left of tau
        mu_loc_l = tau - sigma
        density_l = lambda t, m=mu_loc_l, s=sigma: _gaussian_density(t, m, s, D)
        res_l = compute_discrepancy_fast(density_l, tau, D, grid)
        discs_left.append(res_l["discrepancy"])

    discs_right = np.array(discs_right)
    discs_left  = np.array(discs_left)

    # All should be >= D; allow 1e-4 relative tolerance for numerical integration.
    lower_bound_holds = bool(np.all(discs_right >= D - 1e-4) and np.all(discs_left >= D - 1e-4))

    # Minimum as sigma->0 converges to D
    limiting_disc = float(min(discs_right[-1], discs_left[-1]))

    return {
        "sigmas"             : sigmas,
        "discs_concentrated_right": discs_right.tolist(),
        "discs_concentrated_left" : discs_left.tolist(),
        "lower_bound_holds"  : lower_bound_holds,
        "limiting_discrepancy": limiting_disc,
        "infimum_approached" : abs(limiting_disc - D) < 0.1 * D,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tau, D, horizon = 0.5, 1.0, 1.0
    grid = np.linspace(0.0, horizon, 150)

    # Uniform density (total mass D)
    unif_density = lambda t: D / horizon if 0 <= t <= horizon else 0.0
    disc_unif    = compute_discrepancy(unif_density, tau, D, grid)
    print(f"Uniform density discrepancy: {disc_unif:.4f}  (D = {D})")

    # Gaussian near tau
    gauss_density = lambda t: _gaussian_density(t, tau + 0.05, 0.05, D)
    disc_gauss    = compute_discrepancy_fast(gauss_density, tau, D, grid)
    print(f"Gaussian near tau discrepancy: {disc_gauss['discrepancy']:.4f}")

    # Shrinking-interval lower-bound proof (show small eps end: disc -> D)
    lb = analytical_lower_bound(unif_density, tau, D)
    print(f"\nShrinking-interval convergence (small eps -> disc approaches D):")
    for eps, dc in zip(lb["eps"][:5], lb["discrepancy_contribution"][:5]):
        print(f"  eps={eps:.1e}: |mu-D| = {dc:.4f}")
    print(f"  Limiting value as eps->0: {lb['lower_bound']:.4f} = D")

    # Infimum verification
    print("\n=== Infimum verification ===")
    iv = verify_infimum_equals_D(tau, D, horizon, n_grid=200)
    print(f"  Lower bound holds for all candidates: {iv['lower_bound_holds']}")
    print(f"  Limiting discrepancy as sigma->0:      {iv['limiting_discrepancy']:.4f}")
    print(f"  inf approaches D:                      {iv['infimum_approached']}")

    # Optimisation
    print("\n=== Minimisation search ===")
    opt = minimize_discrepancy(tau, D, horizon, n_grid=150)
    print(f"  Best mu_loc    = {opt['optimal_loc']:.4f}")
    print(f"  Best sigma     = {opt['optimal_sigma']:.4f}")
    print(f"  Min discrepancy= {opt['min_discrepancy']:.4f}  (>= D = {D})")
    print(f"  Gap above D    = {opt['gap']:.4f}")
