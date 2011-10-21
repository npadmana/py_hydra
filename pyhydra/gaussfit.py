import nlopt
import numpy as np


class GaussFit :
  def __init__(self, x, y, err=None, npoly=0):
    self.x = x
    self.y = y
    self.npoly = npoly
    self.nparam = 3 + npoly + 1
    if err is None :
      self.err = self.x * 0.0 + 1.0
    else :
      self.err = err


  def chi2(self, ytest):
    chi = (ytest - self.y)/self.err
    return (chi**2).sum()


  def model(self, vec):
    sig = vec[0]
    x0 = vec[1]
    norm = vec[2]
    pp = np.poly1d(vec[3:])

    tmp = (self.x - x0)**2/(2.0*sig**2)
    yy = np.exp(-tmp) * norm + pp(self.x)

    return yy

  def minfunc(self, vec, grad):
    return self.chi2(self.model(vec))


  def optimize(self, frel=1.e-6, fabs=1.e-8):
    opt = nlopt.opt(nlopt.LN_COBYLA, self.nparam)
    opt.set_min_objective(self.minfunc)
    opt.set_ftol_rel(frel)
    opt.set_xtol_rel(frel)
    opt.set_ftol_abs(fabs)
    opt.set_xtol_abs(fabs)

    startvec = np.zeros((self.nparam, ), dtype='f8')
    startvec[0] = 1.0
    startvec[1] = np.mean(self.x)
    startvec[2] = self.y.max()

    self.xopt = opt.optimize(startvec)
    self.fopt =  opt.last_optimize_result()






