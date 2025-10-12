
"""
Qubit trotterization toy: small 1D chain of truncated oscillators (d levels per site).
Hamiltonian H = H_onsite + H_coupling + H_chi
- H_onsite = ω Σ (n_i + 1/2)
- H_coupling = κ Σ q_i q_{i+1}  (nearest neighbors, open boundary)
- H_chi = χ Σ n_i                    (ether-like diagonal phase term)
First-order Trotter step:
U_step = exp(-i Δt H_onsite) * [∏ even bonds exp(-i Δt κ q_i q_{i+1})] * [∏ odd bonds ...] * exp(-i Δt H_chi)
We build small dense matrices for N<=4, d<=8 so everything runs locally without cloud.
"""
import numpy as np

def ladder_ops(d: int):
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n-1, n] = np.sqrt(n)
    adag = a.conj().T
    n_op = adag @ a
    q = (a + adag) / np.sqrt(2.0)
    p = 1j * (adag - a) / np.sqrt(2.0)
    return a, adag, n_op, q, p

def embed(op: np.ndarray, N: int, site: int, d: int) -> np.ndarray:
    I = np.eye(d, dtype=complex)
    mats = [I]*N
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def embed_two(opA: np.ndarray, i: int, opB: np.ndarray, j: int, N: int, d: int) -> np.ndarray:
    I = np.eye(d, dtype=complex)
    mats = []
    for s in range(N):
        if s == i:
            mats.append(opA)
        elif s == j:
            mats.append(opB)
        else:
            mats.append(I)
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def build_hamiltonian(N: int, d: int, omega: float, kappa: float, chi: float):
    a, adag, n_op, q, p = ladder_ops(d)
    H_onsite = np.zeros((d**N, d**N), dtype=complex)
    for i in range(N):
        H_onsite += omega * (embed(n_op, N, i, d) + 0.5 * embed(np.eye(d), N, i, d))
    H_coupling = np.zeros_like(H_onsite)
    for i in range(N-1):
        H_coupling += kappa * embed_two(q, i, q, i+1, N, d)
    H_chi = np.zeros_like(H_onsite)
    for i in range(N):
        H_chi += chi * embed(n_op, N, i, d)
    H_total = H_onsite + H_coupling + H_chi
    return H_onsite, H_coupling, H_chi, H_total

def expm_hermitian(H: np.ndarray, t: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(H)
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def apply_bond_unitary(state: np.ndarray, U_ij: np.ndarray, i: int, j: int, N: int, d: int) -> np.ndarray:
    psi = state.reshape([d]*N)
    axes = list(range(N))
    axes[0], axes[i] = axes[i], axes[0]
    axes[1], axes[j] = axes[j], axes[1]
    psi_perm = np.transpose(psi, axes)
    psi_2 = psi_perm.reshape(d*d, -1)
    psi_2 = U_ij @ psi_2
    psi_perm = psi_2.reshape([d, d] + [d]*(N-2))
    inv_axes = np.argsort(axes)
    psi_out = np.transpose(psi_perm, inv_axes).reshape(d**N)
    return psi_out

def simulate_trotter(N: int, d: int, omega: float, kappa: float, chi: float, T: float, r: int, psi0: np.ndarray):
    H_onsite, H_coupling, H_chi, H_total = build_hamiltonian(N, d, omega, kappa, chi)
    dt = T / r
    U_onsite = expm_hermitian(H_onsite, dt)
    U_chi = expm_hermitian(H_chi, dt)
    even_bonds = [(i, i+1) for i in range(0, N-1, 2)]
    odd_bonds  = [(i, i+1) for i in range(1, N-1, 2)]
    # two-site unitary for q_i q_{i+1}
    a, adag, n_op, q, p = ladder_ops(d)
    H_loc = np.kron(q, q)
    U_bond = expm_hermitian(H_loc, dt * kappa)

    psi = psi0.copy()
    for _ in range(r):
        psi = U_onsite @ psi
        for (i, j) in even_bonds:
            psi = apply_bond_unitary(psi, U_bond, i, j, N, d)
        for (i, j) in odd_bonds:
            psi = apply_bond_unitary(psi, U_bond, i, j, N, d)
        psi = U_chi @ psi

    norm = np.vdot(psi, psi).real
    drift = abs(norm - 1.0)
    return psi, drift, (H_total, dt)

def reference_evolution(H_total: np.ndarray, T: float, psi0: np.ndarray):
    U_ref = expm_hermitian(H_total, T)
    psi_ref = U_ref @ psi0
    return psi_ref

def trotter_error_norm(psi_trot: np.ndarray, psi_ref: np.ndarray) -> float:
    return float(np.linalg.norm(psi_trot - psi_ref))

def make_initial_state(N: int, d: int, excitations: int = 0) -> np.ndarray:
    basis_states = []
    for i in range(N):
        v = np.zeros(d, dtype=complex)
        v[min(excitations, d-1)] = 1.0
        basis_states.append(v)
    psi = basis_states[0]
    for b in basis_states[1:]:
        psi = np.kron(psi, b)
    return psi
