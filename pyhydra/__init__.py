import os.path
import pysao
import ph_utils as utils

def ds9(*args, **kwargs):
  return pysao.ds9(path=os.path.expandvars('$DS9'))


