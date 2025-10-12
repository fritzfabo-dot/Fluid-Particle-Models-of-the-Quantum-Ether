# phase_c/runner.py
from __future__ import annotations
import argparse, os, time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from braket.aws import AwsDevice
from braket.tasks import QuantumTask

from .trotter_circuit import TrotterSpec, build_trotter_circuit
from .io_utils import ensure_dir, write_json, append_index, extract_execution_ms, now_iso

def device_for(backend: str) -> AwsDevice:
    if backend == "SV1":
        arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    elif backend == "TN1":
        arn = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
    elif backend == "DM1":
        arn = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
    else:
        raise ValueError(f"Unknown backend {backend}")
    return AwsDevice(arn)

def s3_folder():
    bucket = os.environ["BRAKET_BUCKET"]
    prefix = "phase-c"
    return (bucket, prefix)

def coerce_shots_for_backend(backend: str, shots: int) -> int:
    """
    SV1: allow 0 for analytic expectations; if >0 we will sample.
    TN1/DM1: require sampling; enforce 1..1000.
    """
    if backend == "SV1":
        return shots
    # TN1/DM1
    if shots <= 0:
        print(f"[note] {backend} requires sampling; forcing shots=1000")
        return 1000
    return min(shots, 1000)

def _ez_from_counts(counts: Dict[str, int], N: int) -> np.ndarray:
    """
    Compute <Z_q> from bitstring counts.
    We take qubit-0 as the rightmost bit (common convention); for symmetric circuits it won't matter.
    """
    total = float(sum(counts.values()))
    if total == 0:
        return np.zeros(N, dtype=float)
    ez = np.zeros(N, dtype=float)
    for bitstr, cnt in counts.items():
        rev = bitstr[::-1]  # q=0 is LSB
        for q in range(N):
            z = 1.0 if rev[q] == "0" else -1.0
            ez[q] += z * cnt
    return ez / total

def compute_proxy_metrics(result, N: int) -> Dict[str, Any]:
    """
    If result.values present (shots==0 path), use them for <Z>.
    Otherwise (shots>0 path), compute <Z> from measurement_counts.
    """
    ez_list: np.ndarray
    vals = getattr(result, "values", None)

    if vals and len(vals) >= N:
        # expectation-only path (SV1, shots=0)
        ez_list = np.array([float(vals[i]) for i in range(N)], dtype=float)
        p_all_zero = "n/a"
        p_top4_sum = "n/a"
    else:
        # sampling path
        counts = getattr(result, "measurement_counts", None) or {}
        ez_list = _ez_from_counts(counts, N)
        total = float(sum(counts.values())) if counts else 0.0
        if total > 0.0:
            zero_str = "0" * N
            p_all_zero = float(counts.get(zero_str, 0) / total)
            top4 = sorted(counts.values(), reverse=True)[:4]
            p_top4_sum = float(sum(top4) / total)
        else:
            p_all_zero = "n/a"
            p_top4_sum = "n/a"

    return {
        "ez_q0": float(ez_list[0]) if N >= 1 else None,
        "ez_qmid": float(ez_list[N // 2]) if N >= 1 else None,
        "ez_qN1": float(ez_list[N - 1]) if N >= 1 else None,
        "z_sum": float(np.sum(ez_list)) if N >= 1 else None,
        "p_all_zero": p_all_zero,
        "p_top4_sum": p_top4_sum,
    }

def main():
    p = argparse.ArgumentParser(description="Phase C Braket Trotter runner")
    p.add_argument("--run-id", required=True)
    p.add_argument("--backend", choices=["SV1","TN1","DM1"], required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--r", type=int, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--omega", type=float, required=True)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--chi", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", choices=["none","depolarizing","amplitude_damping"], default="none")
    p.add_argument("--noise-p", type=float, default=0.0)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR","./runs"))
    args = p.parse_args()

    # backend-specific shots policy
    args.shots = coerce_shots_for_backend(args.backend, args.shots)

    run_dir = Path(args.runs_dir) / args.run_id
    ensure_dir(run_dir)

    # CONFIG
    config: Dict[str, Any] = dict(
        N=args.N, r=args.r, dt=args.dt, omega=args.omega,
        kappa=args.kappa, chi=args.chi, seed=args.seed,
        backend=args.backend, noise=args.noise, noise_p=args.noise_p,
        shots=args.shots, label=args.label, when=now_iso()
    )
    config_path = run_dir / "config.json"
    _ = write_json(config_path, config)

    # CIRCUIT
    spec = TrotterSpec(
        N=args.N, r=args.r, dt=args.dt, omega=args.omega,
        kappa=args.kappa, chi=args.chi, seed=args.seed,
        backend=args.backend, noise=args.noise, noise_p=args.noise_p,
        shots=args.shots, label=args.label
    )
    circuit = build_trotter_circuit(spec, add_result_types=True)

    # SUBMIT
    device = device_for(args.backend)
    s3_dest = s3_folder()
    t0 = time.time()
    task: QuantumTask = device.run(circuit, s3_destination_folder=s3_dest, shots=args.shots)
    result = task.result()
    t1 = time.time()

    # METRICS
    metrics: Dict[str, Any] = {}
    metrics.update(compute_proxy_metrics(result, args.N))
    exec_ms = extract_execution_ms(result)
    metrics["execution_ms"] = exec_ms
    metrics["wall_clock_s_client"] = round(t1 - t0, 6)
    metrics["task_arn"] = getattr(task, "arn", getattr(task, "id", None))
    metrics["device"] = args.backend
    try:
        metrics["gate_count"] = len(circuit.instructions)
    except Exception:
        pass

    metrics_path = run_dir / "metrics.json"
    _ = write_json(metrics_path, metrics)

    # INDEX
    index_row = dict(
        run_id=args.run_id, backend=args.backend,
        config_path=str(config_path),
        metrics_path=str(metrics_path),
        N=args.N, r=args.r, dt=args.dt, omega=args.omega, kappa=args.kappa, chi=args.chi,
        noise=args.noise, noise_p=args.noise_p, shots=args.shots, label=args.label
    )
    append_index(Path(args.runs_dir) / "INDEX.csv", index_row)

    print(f"[OK] run {args.run_id}  backend={args.backend}  task={metrics['task_arn']}")
    if exec_ms is not None:
        print(f"     backend_exec_ms= {exec_ms}")
    print(f"     client_wall_s = {metrics['wall_clock_s_client']}")
    print(f"     ez(q0)={metrics['ez_q0']}, ez(qN-1)={metrics['ez_qN1']}, p_all_zero={metrics['p_all_zero']}")

if __name__ == "__main__":
    main()
