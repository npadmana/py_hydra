import pyhydra
from ROOTpy import *
import pyfits

datadir = '../Oct20/'

nflats = 9
flist1 = ['flat%02i.fits'%ii for ii in range(nflats)]
flist = [datadir + f1 for f1 in flist1]
imarr = None

# Start looping over the files....
ii = 0
for f1 in flist :
  ff = pyfits.open(f1)
  arr = ff[0].data
  ff.close()

  if imarr is None :
    imarr = zeros((arr.shape[0], arr.shape[1], len(flist)), dtype='f8')
  imarr[:,:, ii] = arr[:,:]
  ii += 1




