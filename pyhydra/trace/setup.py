from distutils.core import setup
from distutils.extension import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

incdir = get_numpy_include_dirs() + ["."]

ext_modules = [Extension("_trace_utils", ["_trace_utils.c"], include_dirs=incdir, libraries=["m"])]

setup(
  name = 'Trace Utilities',
  ext_modules = ext_modules, 
)

