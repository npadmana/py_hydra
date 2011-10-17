import numpy as np
from ph_utils import *
import trace.trace_utils as trace
from __init__ import ds9
import cPickle



# This code assumes that the 2D images are [pos, wavelength]
# If the wavelength is not the second index, use imfix
class HydraRun :
  # Define the basic structures that we fill in 
  def __init__(self, name, imfix=None):
    self.name = name
    self.imfix = imfix
    # Also define the savelist
    self.savelist = ['bias', 'flat2d', 'tracelist', 'masterarc']


  def save(self, basename):
    for isave in self.savelist :
      fn = "%s_%s.pickle"%(basename, isave)
      ff = open(fn, "w")
      exec("cPickle.dump(self.%s, ff)"%isave)
      ff.close()


  def load(self, basename):
    for isave in self.savelist :
      fn = "%s_%s.pickle"%(basename, isave)
      ff = open(fn)
      exec("self.%s = cPickle.load(ff)"%isave)
      ff.close()


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


  def generate_traces(self, sigma=2.0, upper=0.1, lower=0.02, eps=0.03, pad=5, rel=True, verbose=True):
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
    self.tracelist = [ll[ii] for ii in isort]


  def plot_traces(self, vv=None, size=1.0):
    """ Requires ds9.

    vv : an already running pysao ds9 instance 
    """
    if vv is None:
      vv = ds9()
    vv.view(self.get_flat2d())
    for ii in self.tracelist :
      # DS9 is 1-indexed
      ii += 1.0
      vv.mark(ii[:,1], ii[:,0], ii[:,0]*0.0 + size)

    return vv


  def set_masterarc(self, flist, mintrace=0, maxtrace=None,  **kwargs):
    """ Generate a master arc. We do this by simple median
    selection.

    mintrace : The minimum trace to consider
    maxtrace : The maximum trace to consider
    """
    # generate the 2D median frame
    self.masterarc = {}
    self.masterarc['flist'] = flist
    tmp = median_combine(flist=flist, **kwargs) 
    tmp = self._imfix(tmp) - self.get_bias()

    # Extract traces
    out, ivar = boxcar_extract(self.tracelist, tmp)
    self.masterarc['2d'] = out

    # Now median these images.....
    self.masterarc['spec'] = np.median(out[mintrace:maxtrace,:], axis=0)


  def __str__(self):
    return 'HydraRun : %s'%self.name



   
