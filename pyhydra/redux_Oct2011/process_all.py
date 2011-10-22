from misc import *
import os.path

# Load in the basic structure
cc = load_hydrarun('test', 'Trace1')

# Get the list of images 
#datadir = '../Oct20/'
#flist1 = ['field-DEEP2-2hr-1%02i.fits'%ii for ii in range(8)]
#flist2 = ['field-DEEP2-23hr-1%02i.fits'%ii for ii in range(8)]

datadir = '../Oct21/'
outdir = 'Oct21/'
flist3 = ['field-DEEP2-23hr-1%02i.fits'%ii for ii in range(7)]
skylist = ['sky-DEEP2-23hr-1%02i.fits'%ii for ii in range(2)]
flist4 = ['field-EBOSS%02i.fits'%ii for ii in range(5)]
skylist2 = ['sky-EBOSS%02i.fits'%ii for ii in range(1)]

flist = []
#flist.extend(flist1)
#flist.extend(flist2)
flist.extend(flist3)
flist.extend(skylist)
flist.extend(flist4)
flist.extend(skylist2)


# Process all the files in flist
for fn in flist :
  outfn = outdir + fn
  outfn_gz = outfn + '.gz'
  if os.path.exists(outfn) or os.path.exists(outfn_gz) :
    print 'Skipping ...',fn
    continue
  out, ivar = cc.process_single_image(datadir + fn)
  hdu0 = pyfits.PrimaryHDU(out)
  hdu1 = pyfits.ImageHDU(ivar)
  hdulist = pyfits.HDUList([hdu0, hdu1])
  hdulist.writeto(outfn)
  os.system('gzip %s'%outfn)
  print 'Completed ....',fn

  



