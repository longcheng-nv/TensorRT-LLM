# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op40 arm registry: arm name -> (package module name, ctor flag dict).

`base` is the byte-frozen PR#16457 @e612fc2f38 vendored package. Variants are
added as new packages under src/ (never by editing gvrpkg40b) and registered
here. Import side effect free; resolution happens in the harness.
"""

ARMS = {
    # name: (module, ctor_flags)
    "base": ("gvrpkg40b.top_k.gvr_topk_decode", {}),
}


def resolve(name):
    mod_name, flags = ARMS[name]
    import importlib
    mod = importlib.import_module(mod_name)
    return mod.GvrTopKKernel, dict(flags)
