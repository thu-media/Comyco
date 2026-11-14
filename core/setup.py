from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name='slcore',
    version='0.1.0',
    ext_modules=[Pybind11Extension('slcore', ['sim.cc'], cxx_std=17, extra_compile_args=['-O3', '-Wall', '-Wextra'])],
    cmdclass={'build_ext': build_ext},
    package_dir={'': '..'},  # --inplace puts the module next to train.py
    zip_safe=False,
)
