import numpy as np
import nlopt
import gsl



class WaveSol :
  def __init__(self, arclines, knownlines, npix, wgood=None):
    """ arclines : detected arclines in pixels
    knownlines : expected lines, in wavelength
    """
    self.arclines = arclines
    self.knownlines = knownlines
    self.wgood = wgood
    self.npix = npix

  def merit(self, arcwave):
    """ Compute the merit function.

    This is defined by the minimum distance between a detected arcline, 
    and a knownline.

    arcwave : arclines in wavelength

    WARNING: The current implementation is an N^2 implementation. Do NOT 
    USE FOR LARGE NUMBERS OF LINES. It should be fine for ~100 lines though.
    """
    diff = np.subtract.outer(arcwave, self.knownlines)
    diff = diff**2
    tmp = diff.min(axis=1)
    return tmp.sum()


  def objective(self, pp, grad):
    """ The objective function. Ignore grad for now"""
    if len(grad) > 0:
      raise RuntimeError,'Do not pass gradient information'
    if self.wgood is None :
      wave = np.poly1d(pp)(self.arclines)
    else :
      wave = np.poly1d(pp)(self.arclines[self.wgood])
    retval = self.merit(wave)
    return retval

  def fit_linear(self, startwave, endwave, frel=1.e-10, fabs=1.e-12):
    # Set up the optimizer
    opt = nlopt.opt(nlopt.GN_DIRECT, 2)
    opt.set_min_objective(self.objective)
    
    # Set the tolerances
    opt.set_xtol_rel(frel)
    opt.set_ftol_rel(frel)
    opt.set_xtol_abs(fabs)
    opt.set_ftol_abs(fabs)

    # Set bounds
    slope = (endwave - startwave)/np.double(self.npix)
    tmp = np.zeros((2,2), dtype='f8')
    tmp[:,0] = [slope*0.5, startwave*0.5]
    tmp[:,1] = [slope*1.5, startwave*1.5]
    opt.set_lower_bounds(tmp.min(axis=1))
    opt.set_upper_bounds(tmp.max(axis=1))

    startvec = [slope, startwave]
    self.linear = opt.optimize(startvec)


  def idarc(self, pp):
    p1 = np.poly1d(pp)
    arcwave = p1(self.arclines)
    diff = np.subtract.outer(arcwave, self.knownlines)
    diff = np.absolute(diff)
    lineid = diff.argmin(axis=1)
    linedist = diff.min(axis=1)

    return (self.arclines, self.knownlines[lineid], linedist)
  


  def polish(self, niter=None, nk=4, nbreak=10):
    if niter is None :
      niter = [5.0]

    pix, wave, dist = self.idarc(self.linear)
    xev = np.arange(self.npix+1)
    wave0 = self.linear[-1]
    wave1 = np.poly1d(self.linear)(self.npix)


    for wcut in niter :
      ww = np.nonzero(dist < wcut)
      pix = pix[ww]
      wave = wave[ww]
      print 'Using %i points with a cut at %f'%(len(ww[0]), wcut)

      # Assemble the array -- put in start and end points to prevent odd behavious
      xtmp = [0]
      ytmp = [wave0]
      xtmp.extend(pix)
      ytmp.extend(wave)
      xtmp.append(self.npix)
      ytmp.append(wave1)

      yev = gsl.BSplineUniformInterpolate(xtmp, ytmp, xev, xr=[-1.e-4, self.npix+1.e-4], nk=nk, nbreak=nbreak)
      sp = gsl.Spline(xev, yev, 'cspline')

      dist = np.absolute(wave - sp(pix))


    return (xev, yev, pix, wave, sp(pix))


  



