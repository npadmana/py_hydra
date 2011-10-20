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


def medfilt(x=None,L=None):

    '''
    a simple median filter for 1d numpy arrays.

    performs a discrete one-dimensional median filter with window
    length L to input vector x. produces a vector the same size 
    as x. boundaries handled by shrinking L at edges; no data
    outside of x used in producing the median filtered output.
    (upon error or exception, returns None.)

    inputs:
        x, Python 1d list or tuple or Numpy array
        L, median filter window length
    output:
        xout, Numpy 1d array of median filtered result; same size as x
    
    bdj, 5-jun-2009

    This is potentially slow, and can be sped up if necessary.
    '''

    # input checks and adjustments --------------------------------------------
    try:
        N = len(x)
        if N < 2:
            print 'Error: input sequence too short: length =',N
            return None
        elif L < 2:
            print 'Error: input filter window length too short: L =',L
            return None
        elif L > N:
            print 'Error: input filter window length too long: L = %d, len(x) = %d'%(L,N)
            return None
    except:
        print 'Exception: input data must be a sequence'
        return None

    xin = np.array(x)
    if xin.ndim != 1:
        print 'Error: input sequence has to be 1d: ndim =',xin.ndim
        return None
    
    xout = np.zeros(xin.size)

    # ensure L is odd integer so median requires no interpolation
    L = int(L)
    if L%2 == 0: # if even, make odd
        L += 1 
    else: # already odd
        pass 
    Lwing = (L-1)/2

    # body --------------------------------------------------------------------

    for i,xi in enumerate(xin):
  
        # left boundary (Lwing terms)
        if i < Lwing:
            xout[i] = np.median(xin[0:i+Lwing+1]) # (0 to i+Lwing)

        # right boundary (Lwing terms)
        elif i >= N - Lwing:
            xout[i] = np.median(xin[i-Lwing:N]) # (i-Lwing to N-1)
            
        # middle (N - 2*Lwing terms; input vector and filter window overlap completely)
        else:
            xout[i] = np.median(xin[i-Lwing:i+Lwing+1]) # (i-Lwing to i+Lwing)

    return xout

