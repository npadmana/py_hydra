""" Simple plotting routine.

"""

import matplotlib.pyplot as plt
import numpy as np

disp = (7750.11-10372.92)/(4046.13-106.53)

def plotspec(arr):
  ntrace = arr.shape[0]


  ii = 0
  doLoop = True
  while doLoop :
    plt.clf()

    # Read out the spectrum, and get some reasonable limits
    spec = arr[ii, :]
    wave = np.arange(spec.size)
    wave = (wave-106.53) * disp + 10372.92
    med = np.median(spec)
    mad = np.median(np.absolute(spec - med))

    plt.plot(wave,spec)
    plt.ylim(med-10*mad, med+10*mad)

    print 'Trace : %02i, MAD=%f'%(ii, mad)
    
    sel = raw_input()
    if sel == 'q' :
      doLoop = False
    elif sel == 'n' :
      ii = (ii+1)%ntrace
    elif sel == 'p' :
      ii = (ii-1 + ntrace)%ntrace
    
