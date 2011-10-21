import pyhydra
from ROOTpy import *

datadir='../raw/'

overscan = pyhydra.utils.OverScan_Trim(2067, 2132)

cc = pyhydra.HydraRun('test', imfix=overscan) 

# Read in biases 
flist = [datadir + 'bias%02i.fits'%ii for ii in range(1,11)]
cc.set_bias(flist)


# Read in 2d flats
flist = [datadir + 'flat%02i.fits'%ii for ii in range(1,11)]
cc.set_flat2d(flist)

# Generate traces
# Sigma should be greater than the width/sqrt(3)
cc.generate_traces(sigma=2.0) 

# Trace out the flats
cc.trace_flat1d()


# Load in the arc solutions
flist = [datadir + ff for ff in ['f1comp01.fits','f1comp02.fits']]
cc.set_masterarc(flist, mintrace=20, maxtrace=60)
cc.find_masterarc_lines(lo=-50.0)

plt.clf()
cc.plot_masterarc_lines(rcut=None)
plt.savefig('bb_masterarc_all_lines.png')

plt.clf()
cc.plot_masterarc_lines(rcut=-100.0)
plt.savefig('bb_masterarc_rcut100_lines.png')

# Now generate the wavelength solution
arclines = pyhydra.arclines('ThAr', minwave=6000.0, maxwave=11000.0)

startwave = 10000.0 # What is your starting wavelength, guess here
disp = -0.7 # What is your starting dispersion in Ang/pixel, guess here
guess_quality = 0.9 # How good is your guess -- the worse this number is, the longer this will take
rcut = -50.0 # Set some line quality
sigrej= 5.0 # Throw out lines who are sigrej*MAD from zero. This is necessary for robustness
cc.get_wavesol(arclines, rcut, startwave, disp, guess_quality=guess_quality, sigrej=sigrej, niter=[20,10,5]) 

# Plot it
plt.clf()
cc.plot_wavesol(resid=False)
plt.savefig('geha_wavesol.png')
plt.clf()
cc.plot_wavesol(resid=True)
plt.savefig('geha_wavesol_resid.png')



cc.save('test1')

vv = cc.plot_traces()

