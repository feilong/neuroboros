import sys

from setuptools import Extension, setup

# The Voronoi kernel must round every operation on its own, as NumPy does, so
# fused multiply-add is turned off. The kernel is optional: if it cannot be
# built, the package is installed without it and uses NumPy.
if sys.platform == "win32":
    kernel_compile_args = ["/O2", "/fp:precise"]
else:
    kernel_compile_args = ["-O3", "-ffp-contract=off"]

setup(
    name="neuroboros",
    scripts=["bin/npls", "bin/rmdirs", "bin/nbdiff", "bin/nbls", "bin/fs2obj"],
    ext_modules=[
        Extension(
            "neuroboros.surface._voronoi_kernel",
            sources=["src/neuroboros/surface/_voronoi_kernel.c"],
            extra_compile_args=kernel_compile_args,
            optional=True,
        )
    ],
)
