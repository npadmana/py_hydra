""" This is to get me well oriented with the method, none of this is
production, or even development code.


Nikhil Padmanabhan
"""


import ROOTpy as R


def determine_oned_line_points(r1, r2, lo=0.05, eps=0.01):
  r2min = r2.min()
  w1 =  R.nonzero(r2 < (lo*r2min))
  xx = -r1[w1]/r2[w1]
  w2 = R.nonzero(R.absolute(xx) < (0.5+eps)) # Put in an epsilon correction
  return (w1[0])[w2]




def generate_simple_line(err=1.0, h=10.0):
  xx = R.random.randn(1000)*err
  ii = R.random.randint(2, 997)
  print ii
  xx[ii-2 : ii+3] += h
  return xx


def generate_gaussian_line(err=1.0, h=10.0):
  xx = R.random.randn(1000)*err
  ii = R.random.randint(2, 997)
  print ii
  xc = R.arange(1000)
  eterm = -((xc-ii)**2)/(2.0*(2.5**2))
  eterm = R.clip(eterm, -10.0, 0.0)
  xx += R.exp(eterm)*h
  return xx


def gauss(x, sigma):
  # Do all the derivatives at the same time
  eterm = -(x**2)/(2.0*sigma**2)
  eterm = R.clip(eterm, -10.0, 0.0) # Avoid nastiness

  invs2pi = 1./R.sqrt(2.0*R.pi)
  norm = invs2pi * (1./sigma)
  d0 = R.exp(eterm)*norm
  d1 = -x * d0/sigma**2
  
  return (d0, d1)



def generate_derivatives(xx, sigma=1.0):

  # First set up the convolution dimensions
  npix = xx.shape[0]
  npix_full = npix*2
  xx1 = R.zeros((npix_full, ), dtype='f8')
  xx1[0:npix] = xx[:] # Copy into padded array

  # Set up the coordinate arrays
  # We can happily misuse fftfreq here, for simplicity
  xc = R.fft.fftfreq(npix_full, 1./npix_full)


  # Ok, compute first and second derivative kernels
  # Note, this is not optimized for memory etc...
  d0p, d1p = gauss(xc+0.5, sigma)
  d0m, d1m = gauss(xc-0.5, sigma)
  kernel1 = d0p - d0m
  kernel2 = d1p - d1m

  # FFT everything in sight
  xx1_ft = R.fft.rfft(xx1)
  kernel1_ft = R.fft.rfft(kernel1)
  kernel2_ft = R.fft.rfft(kernel2)

  #
  r1 = R.fft.irfft(xx1_ft* kernel1_ft)
  r2 = R.fft.irfft(xx1_ft* kernel2_ft)

  return r1[0:npix], r2[0:npix]



