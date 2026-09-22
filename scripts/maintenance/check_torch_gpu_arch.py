#!/usr/bin/env python3
"""Fail fast when PyTorch CUDA kernels do not include this node's GPU architecture.

Jean Zay V100 = compute capability **7.0** (``sm_70``). Some pip/uv PyTorch wheels
are built only for **7.5+**, which triggers at runtime::

    torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device

Use a PyTorch build that lists ``sm_70`` in ``torch.cuda.get_arch_list()``, or the
cluster's supported ``module load …`` PyTorch stack (see IDRIS Jean Zay GPU docs).

With **uv**, use ``uv sync`` from the repo root (after ``deactivate``/``conda deactivate``
if you had the read-only Jean Zay ``python-3.12.7`` module loaded). PyTorch is pinned via
``pyproject.toml`` and the ``pytorch-cu124`` index; change the index there if needed.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("check_torch_gpu_arch: torch not installed — skip", file=sys.stderr)
        return 0

    if not torch.cuda.is_available():
        print("check_torch_gpu_arch: no CUDA — skip")
        return 0

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    need = f"sm_{major}{minor}"

    arch_list: list[str] = []
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except Exception as exc:
        print(f"check_torch_gpu_arch: could not read get_arch_list(): {exc}", file=sys.stderr)
        return 0

    if not arch_list:
        print("check_torch_gpu_arch: empty get_arch_list() — skip check")
        return 0

    if need in arch_list:
        print(f"check_torch_gpu_arch: OK ({name}, {need} in {arch_list[:5]}…)")
        return 0

    # Some builds use padded names; accept explicit sm_70 for CC 7.0
    if major == 7 and minor == 0 and any("70" in a for a in arch_list if "sm_" in a):
        print(f"check_torch_gpu_arch: OK ({name}, sm_70-compatible tags in arch list)")
        return 0

    print(
        f"check_torch_gpu_arch: ERROR — GPU {name!r} needs {need}, but this PyTorch "
        f"was built only for: {arch_list}\n"
        f"  torch.__version__={torch.__version__}  cuda={torch.version.cuda}\n"
        f"  Fix: reinstall torch with a wheel that includes sm_70, or use Jean Zay's "
        f"recommended GPU Python/PyTorch module instead of a minimal venv wheel.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
