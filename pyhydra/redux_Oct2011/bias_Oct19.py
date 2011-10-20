""" Generate a basic bias frame, and save it.
"""

import pyhydra
from misc import *


datadir='../Oct19/'
nbias = 10

cc = pyhydra.HydraRun('bias_Oct19', imfix=overscan)
flist = [datadir + 'bias%02i.fits'%ii for ii in range(nbias)]
cc.set_bias(flist)

# Save this
cc.save('BiasOct19')

