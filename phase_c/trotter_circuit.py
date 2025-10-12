# phase_c/trotter_circuit.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
from braket.circuits import Circuit, Gate, Observable, Noise, ResultType

NoiseKind = Literal["none", "depolarizing", "amplitude_damping"]

@dataclass
class TrotterSpec:
    N: int
    r: int
    dt: float
    omega: float
    kappa: float
    chi: float
    seed: int
    backend: Literal["SV1","TN1","DM1"]
    noise: NoiseKind = "none"
    noise_p: float = 0.0
    shots: int = 0
    label: str = ""

def _one_trotter_layer(N:int, dt:float, omega:float, kappa:float, chi:float) -> Circuit:
    """
    First-order Trotter for H = H_E + H_B + H_chi:
      H_E  = sum_j omega X_j
      H_B  = sum_{j} kappa Z_j Z_{j+1}
      H_chi= chi * sum_j Z_j
    """
    c = Circuit()

    # e^{-i H_E dt} -> RX(2*omega*dt)
    theta_x = 2.0 * omega * dt
    if abs(theta_x) > 0:
        for q in range(N):
            c.rx(q, theta_x)

    # e^{-i H_B dt} -> CX-RZ-CX for neighbors
    theta_zz = 2.0 * kappa * dt
    for q in range(N - 1):
        c.cnot(q, q + 1)
        c.rz(q + 1, theta_zz)
        c.cnot(q, q + 1)

    # e^{-i H_chi dt} -> RZ(2*chi*dt)
    theta_z = 2.0 * chi * dt
    if abs(theta_z) > 0:
        for q in range(N):
            c.rz(q, theta_z)

    return c

def _apply_noise_global(c: Circuit, spec: TrotterSpec) -> None:
    """
    Attach gate noise across the circuit using apply_gate_noise.
    """
    if spec.noise == "none" or spec.noise_p <= 0:
        return

    if spec.noise == "depolarizing":
        nz = Noise.Depolarizing(probability=spec.noise_p)
    elif spec.noise == "amplitude_damping":
        nz = Noise.AmplitudeDamping(gamma=spec.noise_p)
    else:
        raise ValueError(f"Unknown noise kind: {spec.noise}")

    c.apply_gate_noise(
        nz,
        target_gates=[Gate.Rx, Gate.Rz, Gate.CNot],
        target_qubits=list(range(spec.N)),
    )

def build_trotter_circuit(spec: TrotterSpec, add_result_types: bool = True) -> Circuit:
    """
    Build the r-step circuit.

    Policy to avoid conflicts:
      - shots == 0  => add ResultType.Expectation only (no measurements)
      - shots  > 0  => add measurements only (no result types)
    """
    np.random.seed(spec.seed)

    c = Circuit()
    layer = _one_trotter_layer(spec.N, spec.dt, spec.omega, spec.kappa, spec.chi)
    for _ in range(spec.r):
        c.add(layer)

    _apply_noise_global(c, spec)

    if add_result_types and spec.shots == 0:
        # Expectation values for every qubit; works with SV1 analytic mode
        for q in range(spec.N):
            c.add_result_type(ResultType.Expectation(Observable.Z(), target=q))
    elif spec.shots and spec.shots > 0:
        # Sampling mode: measurements only (no result types allowed)
        for q in range(spec.N):
            c.measure(q)

    return c
