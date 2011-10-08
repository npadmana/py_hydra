import numpy as np
from ph_utils import *

class HydraRun :
  # Define the basic structures that we fill in 
  def __init__(self, name, imfix=None):
    self.name = name
    self.imfix = imfix

  def _imfix(self, arr):
    if self.imfix is None :
      return arr
    else :
      return self.imfix(arr)

  def set_bias_flist(self, flist, **kwargs):
    self.bias = {}
    self.bias['flist'] = flist
    tmp = median_combine(flist=flist, **kwargs)
    self.bias['arr'] = self._imfix(tmp)


  def get_bias(self):
    return self.bias['arr']

  def set_flat2d_flist(self, flist, **kwargs):
    self.flat2d = {}
    self.flat2d['flist'] = flist
    tmp = median_combine(flist=flist, **kwargs) 
    self.flat2d['arr'] = self._imfix(tmp)
    self.flat2d['arr'] -= self.get_bias()


  def get_flat2d(self):
    return self.flat2d['arr']


  def __str__(self):
    return 'HydraRun : %s'%self.name



   
