""" Simple plotting routine.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

disp = (7750.11-10372.92)/(4046.13-106.53)

def plotspec(arr, ivar=None, hydrarun=None, tracenums=None, startpix=400, endpix=4000, ylim=None):
  ntrace = arr.shape[0]
  boxsmooth = 1


  # Get set up for stuff
  if ivar is not None :
    var = 1./(ivar + 1.e-10)

  ii = 0
  doLoop = True
  title = None
  while doLoop :
    plt.clf()

    # Read out the spectrum, and get some reasonable limits
    spec = arr[ii, :]
    wave = np.arange(spec.size)
    wave = (wave-106.53) * disp + 10372.92

    spsmooth = ndimage.uniform_filter1d(spec, boxsmooth, mode='constant')
    plt.plot(wave[startpix:endpix],spsmooth[startpix:endpix])
    if ivar is not None :
      plt.plot(wave[startpix:endpix],np.sqrt(var[ii, startpix:endpix]), 'r-')
      plt.plot(wave[startpix:endpix],-np.sqrt(var[ii, startpix:endpix]), 'r-')
    if ylim is None :
      plt.ylim(-20, 20)
    else :
      plt.ylim(*ylim)

    plt.xlabel('Wavelength')
    plt.ylabel('Counts')
    if title is not None :
      plt.title(title)


    if hydrarun is not None :
      print tracenums[ii], hydrarun.fibermap[tracenums[ii]]



    # Processing loop
    sel = raw_input()
    if sel == 'q' :
      doLoop = False
    elif sel == 't' :
      print 'Enter a title:'
      title = raw_input()
    elif sel == 'n' :
      ii = (ii+1)%ntrace
    elif sel == 'p' :
      ii = (ii-1 + ntrace)%ntrace
    elif sel == 'b' :
      print 'Enter boxcar smoothing in pixels'
      tmp= raw_input()
      boxsmooth = int(tmp)
    elif sel == 'j' :
      print 'Slit # to jump to:'
      tmp = raw_input()
      islit = int(tmp)
      itrace = hydrarun.slit2trace(islit)
      ww = np.nonzero(tracenums==itrace)[0]
      if len(ww) != 0 :
        ii = ww[0]
      else :
        print 'No such slit found...'
    elif sel == 'h' :
      print 'q-quit'
      print 'h-help'
      print 'p-previous spectrum'
      print 'n-next spectrum'
    else :
      ii = (ii+1)%ntrace

