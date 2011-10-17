import pyhydra
import cPickle

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


# Load in the arc solutions
flist1 = loadtxt(datadir+'allarc.plan', dtype=dt)
ww = flist1['y'] == 20.0
flist = [datadir + ff['x'] for ff in flist1[ww]]
cc.set_masterarc(flist, mintrace=20, maxtrace=60)



ff = open('test.pickle','w')
cPickle.dump(cc, ff)
ff.close()


vv = cc.plot_traces()

