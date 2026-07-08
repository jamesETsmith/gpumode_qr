"""Batched square compact-Householder QR benchmark harness (AMD MI350X).

Modules:
- inputs:    problem/input generation (benchmark shapes + stress cases)
- checker:   correctness gates matching the GPUMODE contract
- reference: baseline QR implementations (torch.geqrf)
- bench:     GPU timing harness
- dbwrite:   results DB writer (one JSON per run)
"""

EPS32 = 2.0**-23  # float32 machine epsilon ~1.1920929e-07
