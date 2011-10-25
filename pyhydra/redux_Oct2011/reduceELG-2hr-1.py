# Attempt to do the full reductions on the 23hr DEEP2 field from start to finish
# This combines lots of previous scratch files.... 
# This will also form the basis of codes that I move back to the repo

from ROOTpy import *
import pyhydra
import pyhydra.redux_Oct2011.misc as MM
from pyhydra.ph_utils import saveobj, loadobj
import pyhydra.ph_sky as sky
from matplotlib.backends.backend_pdf import PdfPages
from pyhydra.plotspec import plotspec

cc = MM.load_hydrarun('test', 'Trace1')
startpix = 400
endpix = 4000

rr = {}

# Figure out the good fibers
tmp = loadtxt('BB-ELG-2hr-cut-1-go.hydra.iraf.1', usecols=[0,1])

# Get the good slits
ww = nonzero((tmp[:,1] == 0) | (tmp[:,1] == 1))
slits = tmp[ww[0], 0]
rr['traces'] = array([cc.slit2trace(slit1) for slit1 in slits])
rr['slits'] = slits


# Also get the sky fibers
ww = nonzero(tmp[:,1] == 0)
slits = tmp[ww[0], 0]
rr['skytraces'] = array([cc.slit2trace(slit1) for slit1 in slits])
rr['skyslits'] = slits

flist = []
datadir = 'Oct22/'
nexp = 24
for ii in range(nexp):
  fn = datadir + 'field-ELG-2hr-1%02i.fits.gz'%ii
  flist.append(fn)

nexp = len(flist)
nspec = len(rr['traces'])
rr['flist'] = flist
saveobj('BB-ELG-2hr-cut1-v2.dat', rr)

obj = None
for ii in range(nexp):
  fn = flist[ii]
  ff = pyfits.open(fn)
  arr1 = ff[0].data
  ivar1 = ff[1].data
  ff.close()
  oned, oned_ivar = cc.do_extraction(arr1, ivar1, flatten=True)
  if obj is None :
    nwave = oned.shape[1]
    obj = zeros((nspec, nexp, nwave), dtype='f8')
    obj_ivar = 0.0*obj
  obj[:,ii,:] = oned[rr['traces'],:]
  obj_ivar[:,ii,:] = oned_ivar[rr['traces'], :]
  print 'Processing %s now....'%fn

# Now stack the images
obj_stack, obj_stack_ivar = pyhydra.utils.weighted_average_clip(obj, obj_ivar, axis=1)
rr['obj_stack'] = obj_stack
rr['obj_stack_ivar'] = obj_stack_ivar

# Find a good central sky
# This is only used for alignment
isky = absolute(rr['skytraces'] - 50).argmin()
ispec = nonzero(rr['traces'] == rr['skytraces'][isky])[0]
skyref = obj_stack[ispec[0], :]

obj_clean = obj_stack*0.0
obj_clean_ivar = obj_stack*0.0

# Now continuum subtract 
for ii in range(nspec):
  print 'Median filtering %i spectrum now...'%ii
  tmp, tmp2 = sky.process_one_spec(obj_stack[ii, :], obj_stack_ivar[ii,:], sky0=skyref, maxshift=100)
  obj_clean[ii, :] = tmp
  obj_clean_ivar[ii, :] = tmp2

rr['obj_clean'] = obj_clean
rr['obj_clean_ivar'] = obj_clean_ivar

# Build a reference sky spectrum
##nsky = len(rr['skytraces'])
##skyref = 0.0
##for ii in range(nsky):
##  isky = nonzero(rr['traces'] == rr['skytraces'][ii])[0]
##  skyref += rr['obj_clean'][isky[0],:]

# Try using all the fibers to extract out the sky
skyref = 0.0
for ii in range(nspec):
  skyref += rr['obj_clean'][ii,:]


skyref /= nspec
rr['sky_ref'] = skyref


obj_nosky = rr['obj_clean']*0.0  
fitmeansky = sky.DerivSkySubtract(skyref)
for ii in range(nspec):
  spec, norm,chi2 = fitmeansky.skysubtract(rr['obj_clean'][ii,:], rr['obj_clean_ivar'][ii,:])
  print norm
  obj_nosky[ii,:] = spec

rr['obj_nosky'] = obj_nosky

# Now gear up to do the PCA 
vSVD_full = 0.0*obj_nosky
pcarr = rr['obj_nosky'][:,startpix:endpix]
u, s, vSVD = linalg.svd(pcarr, full_matrices=False)
vSVD_full[:, startpix:endpix] = vSVD

rr['vSVD'] = vSVD_full

skycc= sky.PCASkySubtract(rr['vSVD'], nparam=15, startpix=startpix, endpix=endpix)
obj_nosky_pca = 0.0*obj_nosky
for ii in range(nspec):
  spec, tmp, chi2 = skycc.skysubtract(rr['obj_nosky'][ii,:], rr['obj_stack_ivar'][0,:])
  obj_nosky_pca[ii,:] = spec


rr['obj_nosky_pca'] = obj_nosky_pca
saveobj('BB-ELG-2hr-cut1-v2.dat', rr)





