""" Miscellaneous stuff 


Nikhil Padmanabhan, Yale.
"""

import pyhydra
import pyfits


# Set up the overscan
overscan = pyhydra.utils.OverScan_Trim(2067, 2132)

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
  cc = pyhydra.HydraRun(name, imfix=overscan)
  cc.load(prefix)
  return cc
