#!/usr/bin/env python

from distutils.core import *
from distutils import sysconfig
import os.path

# Get numpy include directory (works across versions)
import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# CCL extension module
_ccllib = Extension(
            "_ccllib",
            ["pyccl/ccl.i",],
            library_dirs=['/usr/local/shared/intel/Compiler/11.1/064/lib/intel64/'],
            #               libraries = ['m', 'gsl', 'gslcblas', 'ccl','fftw3','fftw3_threads','gomp'],
            libraries = ['m', 'gsl', 'gslcblas', 'ccl','fftw3','fftw3_threads','gomp','iomp5'],
            include_dirs = [numpy_include, "include/", "class/include"],
            #               extra_compile_args=['-O4', '-std=c99','-fopenmp'],
            extra_compile_args=['-std=c99','-openmp'],
            swig_opts=['-threads'],
            )

# CCL setup script
setup(  name         = "pyccl",
        description  = "Library of validated cosmological functions.",
        author       = "LSST DESC",
        version      = "0.1",
        packages     = ['pyccl'],
        ext_modules  = [_ccllib,],
        )
