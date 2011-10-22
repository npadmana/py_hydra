""" Miscellaneous stuff 


Nikhil Padmanabhan, Yale.
"""

import pyhydra
import pyfits
from ROOTpy import *


# Set up the overscan
overscan = pyhydra.utils.OverScan_Trim(2067, 2132)
gain_oct2011 = 1./1.4 # e/ADU
readnoise_oct2011 = 3.5 # e

def generate_clean_file(hydrarun, infn, outfn):
  """ A simple utility to read in a file, overscan subtract (image fix),
  bias subtract and write it out as another FITS file.
  """
  # Read in the file
  ff = pyfits.open(infn)
  arr = ff[0].data
  ff.close()

  arrcut = hydrarun._imfix(arr)
  arrcut -= hydrarun.get_bias()


  hdu = pyfits.PrimaryHDU(arrcut)
  hdu.writeto(outfn)




def load_hydrarun(name, prefix='BiasOct19'):
  cc = pyhydra.HydraRun(name, gain_oct2011, readnoise_oct2011, imfix=overscan)
  cc.load(prefix)
  return cc


def process_single_image_foo(infn, hydrarun):
  ff = pyfits.open(infn)
  tmp = ff[0].data
  ff.close()
    
  tmp = hydrarun._imfix(tmp) - hydrarun.get_bias()
  # Extract traces
  out, ivar = pyhydra.boxcar_extract(hydrarun.tracelist, tmp)
  hydrarun.flatten_all(out, nbreak=20)
  return out, ivar


def stack_foo(namelist, hydrarun):
  outarr = None
  ivararr = None

  for ii, fn1 in ndenumerate(namelist) :
    out, ivar = process_single_image_foo(fn1, hydrarun)
    if outarr is None :
      outarr = zeros((out.shape[0], out.shape[1], len(namelist)), dtype='f8')
      ivararr = 0.0*outarr
    outarr[:,:,ii[0]] = out[:,:]
    ivararr[:,:,ii[0]] = ivar[:,:]

  return outarr, ivararr


def print_trace_centroid(hh):
  for ii in range(len(hh.traces['tracelist'])):
    print ii, median(hh.traces['tracelist'][ii][:,0])
  

