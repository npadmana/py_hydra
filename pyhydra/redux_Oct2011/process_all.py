from misc import *
import os.path

# Load in the basic structure
cc = load_hydrarun('test', 'BiasOct19')

# Get the list of images 
datadir = '../Oct20/'
flist1 = ['field-DEEP2-2hr-1%02i.fits'%ii for ii in range(8)]
flist2 = ['field-DEEP2-23hr-1%02i.fits'%ii for ii in range(8)]

flist = []
flist.extend(flist1)
flist.extend(flist2)


# Process all the files in flist
for fn in flist :
  if os.path.exists(fn) :
    print 'Skipping ...',fn
    continue
  out, ivar = cc.process_single_image(datadir + fn)
  hdu0 = pyfits.PrimaryHDU(out)
  hdu1 = pyfits.ImageHDU(ivar)
  hdulist = pyfits.HDUList([hdu0, hdu1])
  hdulist.writeto(fn)
  print 'Completed ....',fn

  



