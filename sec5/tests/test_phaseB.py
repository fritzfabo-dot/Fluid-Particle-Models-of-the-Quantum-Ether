
import numpy as np
from sec5.src.cv_phase import mzi_output_means, ideal_mzi_output_means, small_phi_residual, compute_C1
from sec5.src.trotter_chain import simulate_trotter, reference_evolution, build_hamiltonian, make_initial_state, trotter_error_norm

def test_cv_small_phi_expansion():
    alpha = 0.8 + 0.0j
    phi = 1e-3
    n_max = 40
    res = small_phi_residual(alpha, phi, n_max)
    assert res < 1e-6

def test_cv_truncation_convergence():
    alpha = 1.0 + 0.0j
    phi = 0.5
    n10 = mzi_output_means(alpha, phi, n_max=10)[0]
    n60 = mzi_output_means(alpha, phi, n_max=60)[0]
    n_true = ideal_mzi_output_means(alpha, phi)[0]
    err10 = abs(n10 - n_true)
    err60 = abs(n60 - n_true)
    assert err60 < err10 and err60 / (abs(n_true)+1e-16) < 1e-3

def test_vacuum_limit_identity_phase():
    n0a, n1a = mzi_output_means(0.0+0.0j, 0.0, n_max=20)
    n0b, n1b = mzi_output_means(0.0+0.0j, 1.23, n_max=20)
    assert abs(n0a - n0b) < 1e-12 and abs(n1a - n1b) < 1e-12

def test_C1_dependency_ng_minus_n0():
    eps1_mu1 = 1.2e-18
    n0 = 1.0001
    omega = 1.0e15
    dn0_domega = 2.0e-18
    C1 = compute_C1(eps1_mu1, n0, dn0_domega, omega)
    ng = n0 + omega * dn0_domega
    expected = (eps1_mu1 / (n0**3)) * (ng - n0)
    assert abs(C1 - expected) < 1e-24

def test_trotter_error_halving_with_steps():
    N, d = 4, 4
    omega, kappa, chi = 1.0, 0.1, 1e-3
    T = 0.2
    psi0 = make_initial_state(N, d, excitations=1)
    for r in [10, 20]:
        psi_trot, drift, (H_total, dt) = simulate_trotter(N, d, omega, kappa, chi, T, r, psi0)
        psi_ref = reference_evolution(H_total, T, psi0)
        err = trotter_error_norm(psi_trot, psi_ref)
        if r == 10:
            err10 = err
        else:
            err20 = err
        assert drift < 1e-6
    ratio = err20 / err10
    assert 0.35 < ratio < 0.7
