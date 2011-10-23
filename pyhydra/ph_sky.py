import numpy as np
import gsl
import utils

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
  print ndx.size, L, arr.size, fac 
  return (arr[ndx]).copy()


def matchspec(x, y, npad=None, verbose=True):
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
  iroll = xcorr.argmax()
  
  if iroll > nx1/2 :
    iroll = iroll-nx1+1

  if verbose :
    print 'Shift of %i pixels detected....'%iroll

  return np.roll(y1, iroll)[0:nx]


def process_one_spec(spec, window=100, dilate_fac=20, sky0=None, npad=1000):
  """ Do all the basic processing on a single spectrum 
  
  -- continuum subtraction
  -- matching to sky, if sky0 is not None 
  """
  cont = utils.minfilt(spec, window)
  tmp = spec - cont
  if sky0 is not None :
    # Dilatethe spectrum and the sky 
    spec1 = dilate(tmp, dilate_fac)
    sky0_d = dilate(sky0, dilate_fac)
    # Match these objects
    spec1_m = matchspec(sky0_d, spec1, npad=npad)

    # Contract the 
    return contract(spec1_m, dilate_fac) 
  else :
    return tmp

