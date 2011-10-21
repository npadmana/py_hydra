import numpy as np
import matplotlib.pyplot as plt
from ph_utils import *
import trace.trace_utils as trace
from __init__ import ds9
import cPickle
import gsl
from linefind import *
import wavesol
import scipy.ndimage as ndimage
import cosmics
from matplotlib.backends.backend_pdf import PdfPages
from gaussfit import GaussFit



# This code assumes that the 2D images are [pos, wavelength]
# If the wavelength is not the second index, use imfix
class HydraRun :
  # Define the basic structures that we fill in 
  def __init__(self, name, gain, readnoise, imfix=None):
    """ Set the gain and readnoise 
    
    gain is in electrons/ADU
    """
    self.name = name
    self.imfix = imfix
    self.gain = gain
    self.readnoise = readnoise
    # Also define the savelist
    #self.savelist = ['bias', 'flat2d', 'traces', 'flat1d', 'wavesol']
    self.savelist = ['bias', 'flat2d', 'traces']
    for ii in self.savelist :
      exec('self.%s = None'%ii)


  def save(self, basename):
    for isave in self.savelist :
      if eval('self.%s is not None'%isave):
        print 'Saving %s.....'%isave
        fn = "%s_%s.pickle"%(basename, isave)
        ff = open(fn, "wb")
        exec("cPickle.dump(self.%s, ff, protocol=2)"%isave)
        ff.close()
      else :
        print 'Ignoring %s....'%isave


  def load(self, basename):
    for isave in self.savelist :
      fn = "%s_%s.pickle"%(basename, isave)
      print 'Processing %s from %s......'%(isave, fn)
      try :
        ff = open(fn,'rb')
        exec("self.%s = cPickle.load(ff)"%isave)
        ff.close()
      except :
        print 'Unable to load %s....'%isave


  def _imfix(self, arr):
    if self.imfix is None :
      return arr
    else :
      return self.imfix(arr)

  def set_bias(self, flist, **kwargs):
    self.bias = {}
    self.bias['flist'] = flist
    tmp = median_combine(flist=flist, **kwargs)
    self.bias['arr'] = self._imfix(tmp)


  def get_bias(self):
    return self.bias['arr']

  def set_flat2d(self, flist, **kwargs):
    self.flat2d = {}
    self.flat2d['flist'] = flist
    tmp = median_combine(flist=flist, **kwargs) 
    self.flat2d['arr'] = self._imfix(tmp)
    self.flat2d['arr'] -= self.get_bias()


  def get_flat2d(self):
    return self.flat2d['arr']


  def generate_traces(self, sigma=2.0, upper=0.1, lower=0.02, eps=0.03, pad=5, rel=True, verbose=True, 
                      mkplot=True):
    """ The parameters here are those that go into the trace extraction code.

    sigma : expected width of the line
    rel   : Are the upper/lower thresholds set relative to the minimum, or are they absolute
    upper : upper threshold defined for lines [See Steger paper] 
    lower : lower threshold for growing lines [See Steger paper]
    eps   : Allow center to be outside pixel by eps
    pad   : Padding around the edges to remove effects of Gaussian smoothing
    verbose : Verbosity
  
    """
    arr = self.flat2d['arr']
    hess = trace.generate_hessian(arr, sigma, verbose=verbose)
    trace.salient(hess, upper=upper, lower=lower, eps=eps, pad=pad, rel=rel)
    ll = trace.find_all_lines(hess, verbose=verbose)

    # Loop over the lines and numpy-ify the traces for quicker work
    ll = [np.array(t1) for t1 in ll]
    
    # Compute the mean line position 
    pos = np.array([t1[:,0].mean() for t1 in ll])
    # Sort
    isort = pos.argsort()
    
    # Fill in the traces
    tmp_tracelist = [ll[ii] for ii in isort]

    # Plot setup
    if mkplot :
      plotfn = '%s_trace_draw_qa.pdf'%self.name
      plotcc = PdfPages(plotfn)

    # Now we clean these up
    self.traces = {}
    tracelist = []
    for ii in range(len(tmp_tracelist)):
      itrace = tmp_tracelist[ii]
      minwave = np.round(itrace[:,1].min())
      maxwave = np.round(itrace[:,1].max())
      yy = np.arange(minwave, maxwave+1, dtype='f8')
      xx = gsl.BSplineUniformInterpolate(itrace[:,1], itrace[:,0], yy, xr=[minwave-1., maxwave+1.], nk=4, nbreak=25)
      tmp1 = np.zeros((yy.size,2), dtype='f8')
      tmp1[:,0] = xx
      tmp1[:,1] = yy
      tracelist.append(tmp1)

      if mkplot :
        plt.clf()
        plt.plot(itrace[:,1], itrace[:,0], 'ro')
        plt.plot(yy,xx,'b-')
        plotcc.savefig()
    # Clean up 
    if mkplot :
      plotcc.close()

    # Cache traces
    self.traces['tracelist'] = tracelist


  def set_tracewidths(self, mkplot=True, npix=5):
    # Start by doing a simple box car extraction
    tlist = self.get_tracelist()
    ntrace = len(tlist)
    arr = self.get_flat2d()
    out, outivar = boxcar_cutout(tlist, arr, npix=npix)

    # Do a crude normalization
    norm = out.sum(axis=1)
    out1 = out/(norm[:,np.newaxis,:]+1.e-10)

    # Genetaye profiles
    csec, qaz = weighted_average_clip(out1, outivar)

    widths = []
    if mkplot :
      plotfn = '%s_trace_width_qa.pdf'%self.name
      plotcc = PdfPages(plotfn)
    for ii in range(ntrace):
      # Do the actual fits
      fitcc = GaussFit(np.arange(2*npix+1), csec[ii,:]) 
      fitcc.optimize(frel=1.e-4, fabs=1.e-6)
      widths.append(fitcc.xopt[0])

      if mkplot :
        plt.clf()
        plt.plot(csec[ii, :])
        plt.plot(fitcc.model(fitcc.xopt), 'ro')
        plotcc.savefig()

    # Clean up    
    plotcc.close()
    self.traces['widths'] = widths

  def get_tracelist(self):
    return self.traces['tracelist']

  def plot_traces(self, vv=None, size=1.0):
    """ Requires ds9.

    vv : an already running pysao ds9 instance 
    """
    if vv is None:
      vv = ds9()
    vv.view(self.get_flat2d())
    tracelist = self.get_tracelist()
    for ii in tracelist :
      # DS9 is 1-indexed
      ii += 1.0
      vv.mark(ii[:,1], ii[:,0], ii[:,0]*0.0 + size)
    return vv


###  
###  def trace_flat1d(self):
###    arr = self.get_flat2d()
###    out, ivar = boxcar_extract(self.tracelist, arr)
###    self.flat1d = {}
###    self.flat1d['arr'] = out
###    self.flat1d['ivar'] = ivar
###
###    # Also compute and store the median
###    ww = np.nonzero(ivar > 0.0)
###    self.flat1d['norm'] = np.median(out[ww])
###
###
###  def generate_smooth_flat(self, itrace, nk=4, nbreak=10):
###    # Check basic info
###    ntrace, nwave = self.flat1d['arr'].shape
###    if itrace >= ntrace :
###      raise ValueError, 'No such trace available'
###
###    # Extract the data
###    xx = np.nonzero(self.flat1d['ivar'][itrace,:] > 0.0)[0]
###    yy = self.flat1d['arr'][itrace, xx]/self.flat1d['norm']
###    xev = np.arange(nwave)
###
###    # Fit a B-spline
###    yev = gsl.BSplineUniformInterpolate(xx, yy, xev, xr=[-1.e-4, nwave+1.e-4], nk=nk, nbreak=nbreak)
###    return yev
###
###
###  def flatten_all(self, arr, **kwargs):
###    ntrace = arr.shape[0]
###    for ii in range(ntrace):
###      yev = self.generate_smooth_flat(ii, **kwargs)
###      arr[ii, :] /= yev
###
###  def set_masterarc(self, flist, mintrace=0, maxtrace=None,  nbreak=20, **kwargs):
###    """ Generate a master arc. We do this by simple median
###    selection.
###
###    mintrace : The minimum trace to consider
###    maxtrace : The maximum trace to consider
###    """
###    # generate the 2D median frame
###    self.masterarc = {}
###    self.masterarc['flist'] = flist
###    tmp = median_combine(flist=flist, **kwargs) 
###    tmp = self._imfix(tmp) - self.get_bias()
###
###    # Extract traces
###    out, ivar = boxcar_extract(self.tracelist, tmp)
###    self.flatten_all(out, nbreak=nbreak)
###    self.masterarc['2d'] = out
###
###    # Now median these images.....
###    self.masterarc['spec'] = np.median(out[mintrace:maxtrace,:], axis=0)
###
###
###  def find_masterarc_lines(self, sigma=3.0, lo=-50.0, eps=0.01):
###    xpos, rval = crude_find_lines(self.masterarc['spec'], sigma=sigma, eps=eps, lo=lo)
###    self.masterarc['lines_xpos'] = xpos
###    self.masterarc['lines_rval'] = rval
###    print '%i lines found.....'%len(xpos)
###
###  def plot_masterarc_lines(self, rcut=None):
###    plt.clf()
###    yy = self.masterarc['spec']
###    yymax = yy.max()
###    plt.plot(yy, 'b-')
###    if rcut is None :
###      xpos = self.masterarc['lines_xpos']
###    else :
###      ww = np.nonzero(self.masterarc['lines_rval'] < rcut)
###      xpos = self.masterarc['lines_xpos'][ww]
###    for xx in xpos :
###      plt.plot([xx,xx],[0,yymax], 'r--')
###
###  
###  def get_wavesol(self, knownlines, rcut, startwave, disp, guess_quality=0.9, 
###                 sigrej=5.0, niter=None, nk=4, nbreak=10):
###    self.wavesol = {}
###    self.wavesol['knownlines'] = knownlines
###    if niter is None :
###      niter = [5]
###
###    # Set up
###    npix = len(self.masterarc['spec'])
###    wgood = np.nonzero(self.masterarc['lines_rval'] < rcut)
###
###    # Build class
###    cc = wavesol.WaveSol(self.masterarc['lines_xpos'], knownlines, npix, wgood=wgood, sigrej=sigrej)
###    cc.fit_linear(startwave, disp, guess_quality=guess_quality, wavestep=0.01, dispstep=0.01)
###    print 'The linear first guess fit is ', cc.linear
###    print 'The merit function is ', cc.objective(cc.linear, [])
###    tmp = cc.polish(niter=niter, nk=nk, nbreak=nbreak)
###
###    self.wavesol['lines'] = tmp[2]
###    self.wavesol['waves'] = tmp[3]
###    self.wavesol['predict'] = tmp[4]
###    self.wavesol['xev'] = tmp[0]
###    self.wavesol['yev'] = tmp[1]
###
###
###
###  def plot_wavesol(self, resid=True):
###    if resid :
###      plt.plot(self.wavesol['lines'], self.wavesol['waves']-self.wavesol['predict'], 'ro')
###    else :
###      plt.plot(self.wavesol['lines'], self.wavesol['waves'], 'ro')
###      plt.plot(self.wavesol['xev'], self.wavesol['yev'], 'b-')
###
###

  def process_single_image(self, infn, nlacosmic=2):
    """ Do all the basic clean up steps for a single image 
    
    Imfix
    Bias subtract.
    Clean up cosmic rays
    Generate a noise map

    nlacosmic -- number of LA Cosmic iterations to do
      -- A small number means that a smaller number of cosmics get removed, but the code
         runs faster.
    """

    # Read in the data
    ff = pyfits.open(infn)
    arr = ff[0].data
    ff.close()

    # Image fix and bias subtract
    arr = self._imfix(arr) - self.get_bias()

  
    # Cosmic ray cleanup
    # The parameters here are hardcoded... in general, we should clean these up somewhat
    cc = cosmics.cosmicsimage(arr, gain=self.gain, readnoise=self.readnoise, objlim=1.5)
    cc.findsatstars() # Mark the saturated stars
    cc.run(nlacosmic)
    arr = cc.cleanarray.copy()

    # Now generate a median filtered image
    m5 = ndimage.filters.median_filter(arr, size=5, mode='mirror')
    m5clipped = m5.clip(min=0.00001) # As we will take the sqrt
    ivar = self.gain**2/(self.gain*m5clipped + self.readnoise*self.readnoise)
    # Now set the bad pixels to zero ivar
    ivar[cc.mask == 1] = 0.0

    return arr, ivar


  def __str__(self):
    return 'HydraRun : %s'%self.name



   
