""" Simple plotting routine.

"""

import matplotlib.pyplot as plt

def plotspec(arr):
  ntrace = arr.shape[0]


  ii = 0
  doLoop = True
  while doLoop :
    plt.clf()
    plt.plot(arr[ii, :])

    print 'Trace : %02i'%ii
    
    sel = raw_input()
    if sel == 'q' :
      doLoop = False
    elif sel == 'n' :
      ii = (ii+1)%ntrace
    elif sel == 'p' :
      ii = (ii-1 + ntrace)%ntrace
    
