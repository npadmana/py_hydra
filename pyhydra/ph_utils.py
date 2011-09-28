""" Utility routines for hydra reductions.

Nikhil Padmanabhan, Yale,
Sep 2011
"""

import numpy as np
import pyfits


def median_combine_fits(flist, hdunum=0):
  """ Median combine the files in flist. 

  flist : list of filenames
  hdunum : hdus to combine [0, Primary]

  """
  comb = None
  ii = 0
  for fn in flist :
    cc = pyfits.open(fn, memmap=True)
    dat = cc[hdunum].data
    if comb is None :
      comb = np.zeros((dat.shape[0], dat.shape[1], len(flist)), dtype='f8')
    comb[:,:,ii] = dat[:,:]
    ii += 1
  return np.median(comb, axis=2)


