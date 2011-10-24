""" Utility routines for hydra reductions.

Nikhil Padmanabhan, Yale,
Sep 2011
"""

import numpy as np
import pyfits
import os.path
import cPickle

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



def boxcar_cutout(tracelist, img, ivar=None, npix=2):
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
  ntotpix = 2*npix + 1
  out = np.zeros((nspec, ntotpix, nwave), dtype='f8')
  outivar = np.zeros((nspec, ntotpix, nwave), dtype='f8')

  # Loop over all the traces
  for ispec in range(nspec):
    t1 = tracelist[ispec]
    pixarr = np.int32(np.round(t1[:, 1]))
    icen = np.int32(np.round(t1[:,0]))
    for ii in range(-npix, npix+1) :
      out[ispec, ii+npix, pixarr] = img[icen + ii, pixarr]
      outivar[ispec, ii+npix, pixarr] = ivar[icen + ii, pixarr]


  return (out, outivar)



def gaussian_extract(tracelist, widths, img, ivar=None, npix=5):
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
    w1 = widths[ispec]
    pixarr = np.int32(np.round(t1[:, 1]))
    cen = t1[:,0]
    icen = np.int32(np.round(cen))
    for ii in range(-npix, npix+1) :
      # Compute the weights
      delta_pix = (icen + ii) - cen
      m1 = np.exp(-delta_pix**2/(2.0*w1**2))

      # Gaussian extraction
      out[ispec, pixarr] += (img[icen + ii, pixarr] * m1 * ivar[icen+ii, pixarr])
      outivar[ispec, pixarr] += (m1*m1 * ivar[icen + ii, pixarr])

  # Normalize
  ww = outivar > 0.0
  out[ww] /= outivar[ww]

  return (out, outivar)


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




class OverScan_Trim:
  """ Simple class to expose a function to trim out overscan regions"""
  def __init__(self, start_overscan, end_overscan, prescan=20, trim=10):
    """ start_overscan -- the first overscan pixel
    end_overscan -- the last overscan pixel
    prescan -- trim out columns at the beginning and end of the image.
    trimrows -- trim rows at the top and bottom of the image.
    """
    self.start_overscan = start_overscan
    self.end_overscan = end_overscan
    self.prescan = prescan
    self.trim = trim


  def __call__(self, arr):
    nx, ny = arr.shape
    nx1 = nx-2*self.trim
    ny1 = ny - (self.end_overscan - self.start_overscan + 1) - 2*self.prescan

    arr1 = np.zeros((nx1, ny1), dtype='f8')
    iy1 = self.start_overscan - self.prescan
    arr1[:,0:iy1] = arr[(self.trim+1):(nx-self.trim+1), (self.prescan):(self.start_overscan)]
    arr1[:,iy1:] = arr[(self.trim+1):(nx-self.trim+1), (self.end_overscan+1):(ny-self.prescan)]

    return arr1



def weighted_average_clip(img, ivar, niter=1, sigclip=5.0, axis=-1, returnmask=False):
  """ Combine images along axis
  with an inverse variance weighted sum.

  It then runs a sigma-clipping for n-iterations, effectively setting
  the ivar of the sigma-clipped points to zero.

  Returns the averaged image and ivar.
  """

  # Start by doing the first pass
  ivar1 = ivar.copy()
  werr = ivar1.sum(axis=axis)
  wgood = werr > 0
  av = (img*ivar1).sum(axis=axis) 
  av[wgood] /= werr[wgood]
  # Now expand the average and werr arrays
  av = np.expand_dims(av, axis=axis)
  werr = np.expand_dims(werr, axis=axis)

  nsamp = img.shape[-1]

  # Now sigma clip
  for ii in range(niter):
    sn = np.absolute((img-av)*np.sqrt(ivar1))
    ww = sn > sigclip
    ivar1[ww] = 0.0

    # Redo
    werr = ivar1.sum(axis=axis)
    wgood = werr > 0
    av = (img*ivar1).sum(axis=axis) 
    av[wgood] /= werr[wgood]
    # Now expand the average and werr arrays
    av = np.expand_dims(av, axis=axis)
    werr = np.expand_dims(werr, axis=axis)


  av = np.squeeze(av)
  werr = np.squeeze(werr)

  if returnmask :
    return av, werr, ivar1
  else :
    return av, werr
    


def loadobj(fn):
  ff=open(fn, 'rb')
  obj = cPickle.load(ff)
  ff.close()
  return obj


def saveobj(fn, obj):
  ff = open(fn, 'wb')
  cPickle.dump(obj, ff, protocol=2)
  ff.close()
