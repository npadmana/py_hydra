import numpy as np
from ph_utils import *

class HydraRun :
  # Define the basic structures that we fill in 
  def __init__(self, name):
    self.name = name

  def set_bias_flist(self, flist, **kwargs):
    self.bias = {}
    self.bias['flist'] = flist
    self.bias['arr'] = median_combine(flist=flist, **kwargs)


  def get_bias(self):
    return self.bias['arr']

  def __str__(self):
    return 'HydraRun : %s'%self.name



   
