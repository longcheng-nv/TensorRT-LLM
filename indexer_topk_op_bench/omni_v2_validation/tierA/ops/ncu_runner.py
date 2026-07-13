"""Minimal runner for ncu_attrib.sh: launch the Triton RMSNorm a few times."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import rmsnorm_triton as m

inputs = m.get_inputs()
for _ in range(3):
    m.kernel_fn(*inputs)
import torch
torch.cuda.synchronize()
