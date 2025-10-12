# phase_c/presets.py
from __future__ import annotations
import os
from typing import Tuple
from braket.aws import AwsDevice

SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
TN1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"

def device_for(backend: str) -> AwsDevice:
    backend = backend.upper()
    if backend == "SV1":
        return AwsDevice(SV1_ARN)
    if backend == "TN1":
        return AwsDevice(TN1_ARN)
    if backend == "DM1":
        return AwsDevice(DM1_ARN)
    raise ValueError(f"Unknown backend {backend}")

def s3_folder() -> tuple[str,str]:
    """
    (bucket, prefix) for Braket task outputs.
    Uses env vars: BRAKET_BUCKET, and a fixed prefix 'phase_c'
    """
    bucket = os.environ.get("BRAKET_BUCKET")
    if not bucket:
        raise RuntimeError("BRAKET_BUCKET not set in environment.")
    return (bucket, "phase_c")
