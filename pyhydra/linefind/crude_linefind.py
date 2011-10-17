""" An adaptation of the 2D line finder for 1D.

Nikhil Padmanabhan
"""
import numpy as np

def determine_oned_line_points(r1, r2, lo, eps=0.01):
  w1 =  np.nonzero(r2 < lo) # Select lines
  rval = r2[w1]
  xx = -r1[w1]/r2[w1]
  w2 = np.absolute(xx) < (0.5+eps) # Put in an epsilon correction

  xpos = (w1[0])[w2] + xx[w2]
  rval = rval[w2]
  return (xpos, rval)

def gauss(x, sigma):
  # Do all the derivatives at the same time
  eterm = -(x**2)/(2.0*sigma**2)
  eterm = np.clip(eterm, -10.0, 0.0) # Avoid nastiness

  invs2pi = 1./np.sqrt(2.0*np.pi)
  norm = invs2pi * (1./sigma)
  d0 = np.exp(eterm)*norm
  d1 = -x * d0/sigma**2
  
  return (d0, d1)

def generate_derivatives(xx, sigma=1.0):

  # First set up the convolution dimensions
  npix = xx.shape[0]
  npix_full = npix*2
  xx1 = np.zeros((npix_full, ), dtype='f8')
  xx1[0:npix] = xx[:] # Copy into padded array

  # Set up the coordinate arrays
  # We can happily misuse fftfreq here, for simplicity
  xc = np.fft.fftfreq(npix_full, 1./npix_full)


  # Ok, compute first and second derivative kernels
  # Note, this is not optimized for memory etc...
  d0p, d1p = gauss(xc+0.5, sigma)
  d0m, d1m = gauss(xc-0.5, sigma)
  kernel1 = d0p - d0m
  kernel2 = d1p - d1m

  # FFT everything in sight
  xx1_ft = np.fft.rfft(xx1)
  kernel1_ft = np.fft.rfft(kernel1)
  kernel2_ft = np.fft.rfft(kernel2)

  #
  r1 = np.fft.irfft(xx1_ft* kernel1_ft)
  r2 = np.fft.irfft(xx1_ft* kernel2_ft)

  return r1[0:npix], r2[0:npix]



def find_lines(arr, sigma=1.0, lo=-200, eps=0.01):
  r1, r2 = generate_derivatives(arr, sigma=sigma)
  xpos, rval = determine_oned_line_points(r1, r2, lo=lo, eps=eps)

  return xpos, rval







