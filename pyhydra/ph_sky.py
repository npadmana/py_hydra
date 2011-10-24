import numpy as np
import gsl
import ph_utils as utils
import scipy.ndimage as ndimage
import nlopt

def dilate(arr, fac=10):
  """ Take an array and dilate it by a factor of fac.

  The output array has size (L-1)*fac + 1
  Uses linear interpolation to connect points
  """
  L = arr.size
  xout = np.arange(fac*(L-1)+1, dtype='f8')/fac
  xin = np.arange(L, dtype='f8')
  yout = gsl.Spline(xin, arr, 'cspline')(xout)
  return yout


def contract(arr, fac=10):
  """ Inverse of the dilate step. Make sure you call it exactly the same....
  """
  # Figure out L 
  L = (arr.size - 1)/fac + 1
  ndx = np.arange(L) * fac
  return (arr[ndx]).copy()


def matchspec(x, y, maxroll, npad=None, verbose=True):
  """ Match spectrum y to spectrum x.

  x and y are assumed to be appropriately sampled...

  This then rolls y -- this might of course result in zeros being appended.

  If npad is None, the padding is twice the previous case
  """
  nx = x.size
  if npad is None :
    nx1 = 2*nx
  else :
    nx1 = nx + npad

  x1 = x.copy()
  y1 = y.copy()
  x1.resize(nx1)
  y1.resize(nx1)

  fx1 = np.fft.rfft(x1)
  fy1 = np.fft.rfft(y1)

  xcorr = np.fft.irfft( fx1*fy1.conjugate())
  
  # Do the forward roll
  if xcorr[0:maxroll].max() >  xcorr[-1:-1-maxroll:-1].max() :
    # Positive roll
    iroll = xcorr[0:maxroll].argmax()
  else :
    # Negative toll
    iroll = -xcorr[-1:-1-maxroll:-1].argmax()-1

  if verbose :
    print 'Shift of %i pixels detected....'%iroll

  return np.roll(y1, iroll)[0:nx]


def process_one_spec(spec, window=200, dilate_fac=20, sky0=None, npad=1000, maxshift=100):
  """ Do all the basic processing on a single spectrum 
  
  -- continuum subtraction
  -- matching to sky, if sky0 is not None 


  """
  cont = ndimage.median_filter(spec, window, mode='mirror')
  tmp = spec - cont
  if sky0 is not None :
    # Dilatethe spectrum and the sky 
    spec1 = dilate(tmp, dilate_fac)
    sky0_d = dilate(sky0, dilate_fac)
    # Match these objects
    spec1_m = matchspec(sky0_d, spec1, maxshift*dilate_fac, npad=npad)

    # Contract the 
    return contract(spec1_m, dilate_fac) 
  else :
    return tmp


class PCASkySubtract :
  def __init__(self, eigenspec, nparam=None, startpix=500, endpix=4000):
    self.eigenspec = np.atleast_2d(eigenspec)
    if nparam is None :
      nparam = eigenspec.shape[0]

    self.data = None
    self.ivar = None
    self.reset_optimizer(nparam)
    self.startpix = startpix
    self.endpix = endpix


  def reset_optimizer(self, nparam):
    self.nparam = nparam
    self.opt = nlopt.opt(nlopt.LN_COBYLA, nparam)
    self.opt.set_ftol_rel(1.e-6)
    self.opt.set_xtol_rel(1.e-8)
    self.opt.set_ftol_abs(1.e-6)
    self.opt.set_xtol_abs(1.e-8)
    self.opt.set_min_objective(self.minfunc)

  def resid(self, vec):
    model = 0*self.data
    for ii in range(self.nparam):
      model += self.eigenspec[ii, :]*vec[ii]
    return self.data - model


  def minfunc(self, vec, grad):
    rr = self.resid(vec)
    chi2 = rr**2 * self.ivar
    return (chi2[self.startpix:self.endpix]).sum()
  


  def skysubtract(self, data, ivar):
    self.data = data
    self.ivar = ivar

    startvec = np.zeros((self.nparam, ), dtype='f8')
    outvec = self.opt.optimize(startvec)
    rr = self.resid(outvec)
    chi2 = self.minfunc(outvec, None)

    return (rr, outvec, chi2)





