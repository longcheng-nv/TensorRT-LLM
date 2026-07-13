# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ncu attribution runner: launch one impl a few times on the TOKENS cell.
Usage: KERNEL_REGEX=... TOKENS=... bash ncu_attrib.sh src/ncu_runner.py <impl.py>"""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("impl", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.get_inputs()
for _ in range(3):
    mod.kernel_fn(*inputs)  # JIT warmup; ncu -k regex keeps these out of the verdict block anyway
torch.cuda.synchronize()
mod.kernel_fn(*inputs)
torch.cuda.synchronize()
