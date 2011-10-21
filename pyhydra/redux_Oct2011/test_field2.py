from misc import *

# Load in the basic structure
cc = load_hydrarun('test', 'TraceOct19')

# Get the list of images 
datadir = '../Oct20/'
flist = [datadir + 'field-DEEP2-2hr-1%02i.fits'%ii for ii in range(5)]


outarr = None
ivararr = None

ii = 0
for fn in flist :
  out, ivar = cc.process_single_image(fn)
  if outarr is None :
    outarr = zeros((out.shape[0], out.shape[1], len(flist)), dtype='f8')
    ivararr = 0.0*outarr
  outarr[:,:,ii] = out[:,:] 
  ivararr[:,:,ii] = ivar[:,:]
  ii += 1


av, av_ivar = pyhydra.utils.weighted_average_clip(img, ivar, sigclip=3.5)


