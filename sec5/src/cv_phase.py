
"""
CV photonics toy: Mach–Zehnder interferometer with an ether-phase U_chi(phi) = exp(i phi n̂)
Modeled with coherent states and truncated Fock expansions to show convergence vs n_max.
No external CV libraries required.
"""
import numpy as np
from math import factorial

def coherent_coeffs(alpha: complex, n_max: int) -> np.ndarray:
    """Return truncated coherent state coefficients c_n up to n_max (unnormalized)."""
    c = np.zeros(n_max+1, dtype=complex)
    for n in range(n_max+1):
        c[n] = alpha**n / np.sqrt(factorial(n))
    return c

def truncated_poisson_mean(beta_abs2: float, n_max: int) -> float:
    """
    Exact mean of a coherent state's photon number with Poisson( |beta|^2 )
    but truncated at n_max with renormalization.
    """
    P = np.array([beta_abs2**n / factorial(n) for n in range(n_max+1)], dtype=float)
    Z = np.sum(P)
    n_vals = np.arange(n_max+1, dtype=float)
    mean_trunc = np.sum(n_vals * P) / Z
    return mean_trunc

def mzi_output_means(alpha: complex, phi: float, n_max: int) -> tuple[float, float]:
    """
    Mach–Zehnder: input |alpha> in port 0, |0> in port 1.
    BS (50/50), phase U_chi(phi)=exp(i phi n̂) on arm a, BS (50/50).
    Coherent-state propagation gives output coherent amplitudes:
      beta0 = (alpha/2) * (1 + e^{i phi})
      beta1 = (alpha/2) * (1 - e^{i phi})
    We compute mean photon numbers from truncated Poisson means to demonstrate convergence.
    """
    beta0 = 0.5 * alpha * (1.0 + np.exp(1j*phi))
    beta1 = 0.5 * alpha * (1.0 - np.exp(1j*phi))
    nbar0_trunc = truncated_poisson_mean(np.abs(beta0)**2, n_max)
    nbar1_trunc = truncated_poisson_mean(np.abs(beta1)**2, n_max)
    return float(nbar0_trunc), float(nbar1_trunc)

def ideal_mzi_output_means(alpha: complex, phi: float) -> tuple[float, float]:
    a2 = abs(alpha)**2
    return 0.5 * a2 * (1.0 + np.cos(phi)), 0.5 * a2 * (1.0 - np.cos(phi))

def small_phi_residual(alpha: complex, phi: float, n_max: int) -> float:
    n0_trunc, _ = mzi_output_means(alpha, phi, n_max)
    approx = abs(alpha)**2 * (1.0 - (phi**2)/4.0)
    return abs(n0_trunc - approx) / (abs(approx) + 1e-18)

def compute_C1(eps1_plus_mu1: float, n0: float, dn0_domega: float, omega: float) -> float:
    ng = n0 + omega * dn0_domega
    return (eps1_plus_mu1 / (n0**3)) * (ng - n0)
