import os.path
import pysao

def ds9(*args, **kwargs):
  return pysao.ds9(path=os.path.expandvars('$DS9'))


