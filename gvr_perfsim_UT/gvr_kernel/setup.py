"""Optional installable build (`pip install -e .`) — alternative to the
JIT load in __init__.py. Either works; JIT is the default path.
"""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_csrc = Path(__file__).parent / "csrc"

setup(
    name="gvr_kernel",
    version="0.1.0",
    description="Standalone DSv4 Pro GVR Heuristic Top-K kernel (Blackwell).",
    ext_modules=[
        CUDAExtension(
            name="gvr_kernel_ext",
            sources=[str(_csrc / "binding.cpp"),
                     str(_csrc / "heuristicTopKDecode.cu")],
            include_dirs=[str(_csrc)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "--use_fast_math", "-std=c++17",
                    "--expt-relaxed-constexpr", "--extended-lambda",
                    "-gencode=arch=compute_100,code=sm_100",
                    "-gencode=arch=compute_103,code=sm_103",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["gvr_kernel"],
    package_dir={"gvr_kernel": "."},
)
