""" Generate the traces and store them in a separate set
"""

from misc import *

cc = load_hydrarun('traces', prefix='Bias')
datadir = '../Oct20/'

flist = [datadir + 'flat-rcirlce%02i.fits'%ii for ii in range(10)]
cc.set_flat2d(flist)

# Generate traces
# Sigma should be greater than the width/sqrt(3)
cc.generate_traces(sigma=2.0) 
cc.save('Trace1')
