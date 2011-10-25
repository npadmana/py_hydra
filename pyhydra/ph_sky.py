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


def matchspec(x, y, yerr, maxroll, npad=None, verbose=True):
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
  yerr1 = yerr.copy()
  x1.resize(nx1)
  y1.resize(nx1)
  yerr1.resize(nx1)

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

  y_s = np.roll(y1, iroll)[0:nx]
  yerr_s = np.roll(yerr1, iroll)[0:nx]


  return y_s, yerr_s


def process_one_spec(spec, err,  window=200, dilate_fac=20, sky0=None, npad=1000, maxshift=100):
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
    err1 = dilate(err, dilate_fac)
    # Match these objects
    spec1_m, err1_m = matchspec(sky0_d, spec1, err1, maxshift*dilate_fac, npad=npad)

    # Contract the 
    s1 = contract(spec1_m, dilate_fac) 
    e1 = contract(err1_m, dilate_fac) 
    return (s1, e1)
  else :
    return (tmp, err) # Pass through the error -- nothing really changed there


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
    self.opt = nlopt.opt(nlopt.LD_TNEWTON, nparam)
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
    if grad is not None :
      if len(grad) > 0 :
        for ii in range(self.nparam):
          tmp = (2.0*rr*self.eigenspec[ii, :] * self.ivar)
          grad[ii] = -tmp[self.startpix:self.endpix].sum()

    return (chi2[self.startpix:self.endpix]).sum()
  


  def skysubtract(self, data, ivar):
    self.data = data
    self.ivar = ivar

    startvec = np.zeros((self.nparam, ), dtype='f8')
    outvec = self.opt.optimize(startvec)
    rr = self.resid(outvec)
    chi2 = self.minfunc(outvec, None)

    return (rr, outvec, chi2)



class DerivSkySubtract :
  def __init__(self, skyspec, polydeg=3, startpix=500, endpix=4000, smooth=1.0):

    skyspec1 = ndimage.gaussian_filter1d(skyspec, smooth, mode='constant')
    self.smooth = 1.0

    # Work on temmplates
    skyp1 = ndimage.shift(skyspec, 0.1)
    skym1 = ndimage.shift(skyspec, -0.1)
    dsky = (skyp1 - skyspec)/0.1
    d2sky = (skyp1 + skym1 - 2.0*skyspec)/(0.1**2)

    # Store templates
    self.templates = [skyspec, dsky, d2sky]
    self.ntemp = len(self.templates)

    self.data = None
    self.ivar = None
    self.startpix = startpix
    self.endpix = endpix
    npix = len(skyspec)
    self.xpix = np.arange(npix, dtype='f8')/npix
    self.polydeg = polydeg # Degree of polynomial
    self.nparam = self.ntemp*(self.polydeg+1) # There are two components for now

    self.opt = nlopt.opt(nlopt.LD_TNEWTON, self.nparam)
    self.opt.set_ftol_rel(1.e-6)
    self.opt.set_xtol_rel(1.e-8)
    self.opt.set_ftol_abs(1.e-6)
    self.opt.set_xtol_abs(1.e-8)
    self.opt.set_min_objective(self.minfunc)

  def resid(self, vec):
    model = 0.0
    for ii in range(self.ntemp):
      pp = np.poly1d(vec[ii*(self.polydeg+1) : (ii+1)*(self.polydeg+1)])
      model = model + pp(self.xpix)* self.templates[ii]
    return self.data - model


  def minfunc(self, vec, grad):
    self.curvec = vec
    rr = self.resid(vec)
    chi2 = rr**2 * self.ivar
    if grad is not None :
      if  len(grad) > 0 :
        for ii in range(self.ntemp):
          tmp1 = 2.0*rr*self.templates[ii]*self.ivar
          for jj in range(self.polydeg+1) :
            tmp = tmp1  * self.xpix**(self.polydeg-jj)
            grad[ii*(self.polydeg+1) + jj] = -tmp[self.startpix:self.endpix].sum()

    return (chi2[self.startpix:self.endpix]).sum()
  
  def skysubtract(self, data, ivar):
    self.data = ndimage.gaussian_filter1d(data, self.smooth, mode='constant')
    self.ivar = ivar


    startvec = np.zeros((self.nparam, ), dtype='f8')
    outvec = self.opt.optimize(startvec)
    rr = self.resid(outvec)
    chi2 = self.minfunc(outvec, None)

    return (rr, outvec, chi2)


class DerivSkySubtract_v2 :
  def __init__(self, skyspec, polydeg=1, startpix=500, endpix=4000):
    self.skyspec = skyspec


    # Work on temmplates
    skyp1 = ndimage.shift(skyspec, 0.1)
    dsky = (skyp1 - skyspec)/0.1

    # Store templates
    self.ntemp = 3

    self.data = None
    self.ivar = None
    self.startpix = startpix
    self.endpix = endpix
    npix = len(skyspec)
    self.xpix = np.arange(npix, dtype='f8')/npix
    self.polydeg = polydeg # Degree of polynomial
    self.nparam = self.ntemp*(self.polydeg+1) + 2 # Add in a smoothing component

    self.opt = nlopt.opt(nlopt.LN_COBYLA, self.nparam)
    self.opt.set_ftol_rel(1.e-3)
    self.opt.set_xtol_rel(1.e-5)
    self.opt.set_ftol_abs(1.e-3)
    self.opt.set_xtol_abs(1.e-5)

    # We need to set maximum and minimum bounds
    lb = np.zeros(self.nparam) - np.inf
    ub = np.zeros(self.nparam) + np.inf
    lb[0:2] = 1.e-3
    ub[0:2] = 1.75
    self.opt.set_lower_bounds(lb)
    self.opt.set_upper_bounds(ub)

    self.opt.set_min_objective(self.minfunc)

  def resid(self, vec):
    model = 0.0
    data1 = ndimage.gaussian_filter1d(self.data, vec[0], mode='constant')
    t0 = ndimage.gaussian_filter(self.skyspec, vec[1])
    t0p = ndimage.shift(t0, 0.1)
    t0m = ndimage.shift(t0, -0.1)
    dsky = (t0p-t0)/0.1
    d2sky = (t0p + t0m - 2*t0)/(0.1**2)
    templates = [t0, dsky, d2sky]

    for ii in range(self.ntemp):
      istart = ii*(self.polydeg+1) + 2
      iend = (ii+1)*(self.polydeg+1) + 2
      pp = np.poly1d(vec[istart : iend])
      model = model + pp(self.xpix)* templates[ii]
    return data1 - model


  def minfunc(self, vec, grad):
    rr = self.resid(vec)
    chi2 = rr**2 * self.ivar
    return (chi2[self.startpix:self.endpix]).sum()
  
  def skysubtract(self, data, ivar):
    self.data = data
    self.ivar = ivar

    startvec = np.zeros((self.nparam, ), dtype='f8')
    startvec[0:2] = 1.0
    outvec = self.opt.optimize(startvec)
    rr = self.resid(outvec)
    chi2 = self.minfunc(outvec, None)

    return (rr, outvec, chi2)






