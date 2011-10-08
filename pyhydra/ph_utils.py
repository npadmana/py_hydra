""" Utility routines for hydra reductions.

Nikhil Padmanabhan, Yale,
Sep 2011
"""

import numpy as np
import pyfits

def load_multiple_fits(flist, hdunum=0, verbose=False, transpose=False, **kwargs):
  """ Read in all the FITS files specified in flist, 
  reading in HDU hdnum, and fill a 
  numpy array with it. 

  Will raise a ValueError if the arrays don't match in size
  """
  comb = None
  ii = 0
  for fn in flist :
    if verbose :
      print "Processing %s now......"%fn
    cc = pyfits.open(fn, memmap=True)
    dat = cc[hdunum].data
    if transpose :
      dat = dat.T.copy() 
    if comb is None :
      comb = np.zeros((dat.shape[0], dat.shape[1], len(flist)), dtype='f8')
      savsize = dat.shape
    if (dat.shape != savsize) :
      raise ValueError, 'File %s has an incorrectly shaped array'%fn
    comb[:,:,ii] = dat[:,:]
    cc.close()
    ii += 1
  return comb



def median_combine(flist=None, arr=None, axis=-1, **kwargs):
  """ Median combine files.

  flist : list of filenames, use load_multiple_fits
  axis : -1
  arr : 

  """
  if flist is not None :
    arr = load_multiple_fits(flist, **kwargs)
  if arr is None :
    raise SyntaxError, 'Specify either flist or arr'
  return np.median(arr, axis=axis)


