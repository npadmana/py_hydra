""" Code to detect, and trace lines in a 2D image.

Based on Steger, An unbiased detector of curvilinear structures, 
IEEE Transactions on Pattern Analysis and Machine Intelligence,
1998 vol. 20 (2) pp. 113 - 125

ASSUMPTIONS :
  1. Positive lines
  2. Does not currently deal with junctions 
This version of the code, as written will only detect positive 
lines. 

Nikhil Padmanabhan, Yale, 2011
"""

import numpy as np
import _trace_utils 

def _gauss(x,y, sigma):
  norm = 1./(2.0*np.pi*sigma**2)
  eterm = (x**2 + y**2)/(2.0*sigma**2)
  eterm = np.clip(eterm, 0.0, 20.0)
  return norm * np.exp(-eterm)

def _gauss_x(x,y, sigma):
  tmp = _gauss(x,y,sigma)
  return -x * tmp/sigma**2


def _gauss_y(x,y,sigma):
  tmp = _gauss(x,y,sigma)
  return -y * tmp/sigma**2


def _generate_gaussian_kernel(x, y, sigma, select):
  """ select is one of :
         x, y, xx, xy, yx, yy
  """
  # Selector
  if select == "x" :
    return _gauss(x+0.5, y, sigma) - _gauss(x-0.5, y, sigma)
  elif select == "y" :
    return _gauss(x, y+0.5, sigma) - _gauss(x, y-0.5, sigma)
  elif select == "xx" :
    return _gauss_x(x+0.5, y, sigma) - _gauss_x(x-0.5, y, sigma)
  elif select == "yy" :
    return _gauss_y(x, y+0.5, sigma) - _gauss_y(x, y-0.5, sigma)
  elif select == "xy" :
    return _gauss_x(x, y+0.5, sigma) - _gauss_x(x, y-0.5, sigma)
  elif select == "yx" :
    return _gauss_y(x+0.5, y, sigma) - _gauss_y(x-0.5, y, sigma)
  else :
    raise RuntimeError, 'Unknown selection'


def generate_hessian(A, sigma, verbose=False):
  """Generate the Hessian matrix
  This returns a [Nx+2, Ny+2, 6] array with the following
  elements
   1 = Ix, 2 = Iy
   2 = Ixx, 3=Iyy, 4 = Ixy, 5 = Iyx
  Note that Ixy = Iyx, but since we calculate them differently, 
  we store both cases.
  The +2 padding (1 around the boundaries) is just to simplify the 
  linking algorithm, so that we don't need to special case edge points

  Syntax :
     generate_hessian(A, sigma)

  A is the array, sigma is the smoothing scale.
  """
  # Get the dimensions of the array
  nx, ny = A.shape

  # Allocate the output
  hess = np.zeros((nx+2, ny+2, 6), dtype='f8')


  # Zero pad A
  A1 = np.zeros((2*nx, 2*ny),dtype='f8')
  A1[0:nx, 0:ny] = A[:,:]
  # FFT
  A1fft = np.fft.rfft2(A1)

  # Define coordinates
  xc = np.fft.fftfreq(2*nx, 1./(2.0*nx))
  yc = np.fft.fftfreq(2*ny, 1./(2.0*ny))
  # Note swapped version of x and y
  ycoord, xcoord = np.meshgrid(yc, xc)

  # Derivs 
  derivlist = ['x','y','xx', 'yy','xy','yx']
  for ii, select1 in np.ndenumerate(derivlist):
    kernel = _generate_gaussian_kernel(xcoord, ycoord, sigma, select1)
    kft = np.fft.rfft2(kernel)
    val = np.fft.irfft2(A1fft*kft)
    hess[1:nx+1, 1:ny+1, ii[0]] = val[0:nx,0:ny] # Remove padding
    if verbose :
      print 'Completed %s'%select1

  # return output 
  return hess


def salient(hess, lower=0.02, upper=0.1, pad=5, eps=0.03, rel=True):
  """
  hess : Hessian structure from generate_hessian
  lower : lower bound -- for line following
          points must have 2nd derivs < lower
  upper :
          upper bound to define salient points, for line starts
          points must have 2nd derivs < upper
  rel : 
        if True, then lower and upper are fractions of the most negative eigenvalue
        and should be positive. Otherwise these are absolute #s.
  pad :
        Padding around edges; no salient points defined here.
  eps :
        allow subgrid center to be outside pixel by eps
  """
  rmin = _trace_utils._do_jacobi2x2_all(hess)

  print "Minimum eigenvalue : %f"%rmin
  if rmin >= 0 :
    raise RuntimeError, 'rmin is not less than zero'

  if rel :
    up = upper * rmin
    low = lower * rmin
  else :
    up = upper
    low = lower

  _trace_utils._saliency(hess, up, low, eps, pad)
  _trace_utils._remove_coincident(hess, eps*2.0)



