import pyhydra
from ROOTpy import *

datadir='../rawdata/jul2010/'

cc = pyhydra.HydraRun('test', imfix=pyhydra.utils.transpImage) 

# Read in biases 
dt = dtype([('x', 'S21'), ('y', 'f8')])
flist1 = loadtxt(datadir+'bias.plan', dtype=dt)
flist = [datadir + ff for ff in flist1['x']]
cc.set_bias(flist)


# Read in 2d flats
dt = dtype([('x', 'S21'), ('y', 'f8')])
flist1 = loadtxt(datadir+'flats.plan', dtype=dt)
flist = [datadir + ff for ff in flist1['x']]
cc.set_flat2d(flist)


# Generate traces
# The defaults have been tuned for this
cc.generate_traces()

# Trace out the flats
cc.trace_flat1d()


# Load in the arc solutions
flist1 = loadtxt(datadir+'allarc.plan', dtype=dt)
ww = flist1['y'] == 20.0
flist = [datadir + ff['x'] for ff in flist1[ww]]
cc.set_masterarc(flist, mintrace=20, maxtrace=60)
cc.find_masterarc_lines(lo=-20.0)

plt.clf()
cc.plot_masterarc_lines(rcut=None)
plt.savefig('geha_masterarc_all_lines.png')

plt.clf()
cc.plot_masterarc_lines(rcut=-100.0)
plt.savefig('geha_masterarc_rcut100_lines.png')

# Now generate the wavelength solution
arclines = pyhydra.arclines('CuAr', minwave=3000.0, maxwave=9000.0)

startwave = 7500.0 # What is your starting wavelength, guess here
disp = -1.5 # What is your starting dispersion in Ang/pixel, guess here
guess_quality = 0.8 # How good is your guess -- the worse this number is, the longer this will take
rcut = -100.0 # Set some line quality
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

