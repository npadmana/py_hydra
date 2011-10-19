""" Utility routines for hydra reductions.

Nikhil Padmanabhan, Yale,
Sep 2011
"""

import numpy as np
import pyfits
import os.path

def load_multiple_fits(flist, hdunum=0, verbose=False, **kwargs):
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


def transpImage(arr):
  return arr.T.copy()


def boxcar_extract(tracelist, img, ivar=None, npix=2):
  """ Simple boxcar extraction.

  Use pixels [-npix, -npix+1, .., 0, .., npix-1, npix]

  If ivar is set to None, artificially set ivar to 1 for all elements.
  This is necessary, since the traces might skip a pixel due to roundoff error.
  """
  if ivar is None :
    ivar = img*0.0 + 1.0

  # Define the output arrays and ivars
  nspec = len(tracelist)
  nwave = img.shape[1]
  out = np.zeros((nspec, nwave), dtype='f8')
  outivar = np.zeros((nspec, nwave), dtype='f8')

  # Loop over all the traces
  for ispec in range(nspec):
    t1 = tracelist[ispec]
    pixarr = np.int32(np.round(t1[:, 1]))
    icen = np.int32(np.round(t1[:,0]))
    for ii in range(-npix, npix+1) :
      out[ispec, pixarr] += img[icen + ii, pixarr]
      outivar[ispec, pixarr] += ivar[icen + ii, pixarr]


  return (out, outivar)



def mk_ivar(img, readnoise=0.0, gain=1.0):
  """ Generate a Poisson noise image.

  """
  # Generate the variance
  var = img*gain + readnoise**2

  # Invert, avoiding divide by zero
  wwvar = var > 0.0
  ivar = 0.0*var
  ivar[wwvar] = 1./var[wwvar]

  return ivar



def arclines(lamptype, minwave, maxwave):
  """ Return a list of lines.

  lamptype :
     ThAr : Thorium-Argon
     HeNeAr : Helium Neon Argon
     CuAr   : Copper-Argon

  These files are in PYHYDRA_DIR/data/

  minwave, maxwave : min, max wavelength
  """
  fndict = {}
  fndict['HeNeAr'] = 'henearhres.dat'
  fndict['CuAr'] = 'cuar.dat'
  fndict['ThAr'] = 'thar.dat'

  try :
    fn = os.path.join(os.path.expandvars("$PYHYDRA_DIR/data"), fndict[lamptype])
  except KeyError :
    raise KeyError, 'Unknown lamptype.'


  arr = np.loadtxt(fn, usecols=[0])
  ww = np.nonzero((arr >= minwave) & (arr <= maxwave))

  return arr[ww]




