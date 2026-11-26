from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildKernel(build_ext):
    """Build the Voronoi kernel without fused multiply-add.

    The kernel must round every operation on its own, as NumPy does, so the
    compiler may not fuse a product and a sum into one operation.
    """

    def build_extension(self, ext):
        if self.compiler.compiler_type == "msvc":
            ext.extra_compile_args = ["/O2", "/fp:precise"]
        else:
            ext.extra_compile_args = ["-O3", "-ffp-contract=off"]
        super().build_extension(ext)


setup(
    name="neuroboros",
    scripts=["bin/npls", "bin/rmdirs", "bin/nbdiff", "bin/nbls", "bin/fs2obj"],
    # The kernel is optional: if it cannot be built, the package is installed
    # without it and uses NumPy.
    ext_modules=[
        Extension(
            "neuroboros.surface._voronoi_kernel",
            sources=["src/neuroboros/surface/_voronoi_kernel.c"],
            optional=True,
        )
    ],
    cmdclass={"build_ext": BuildKernel},
)
