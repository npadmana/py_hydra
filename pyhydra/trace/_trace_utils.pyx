import numpy as np
cimport numpy as np
cimport cython

# Transform the derivative array into a line-point array
# Reuse the data as follows 
#   1 -- max eigenvalue
#   2 -- angle
#   3 -- px 
#   4 -- py
#   5 -- flag : 0 = not a line point  
#               1 = line point for start
#               2 = line start for line
#   6 -- flag : 0 = not assigned
#               1 = assigned

# Get the C-routines for arithmetic
cdef extern from "math.h" :
  double atan(double)
  double sin(double)
  double cos(double)
  double fabs(double)

# Turn off error checking!!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] jacobi2x2(np.ndarray[np.float64_t, ndim=1] arr):
  """ Compute the maximum eigenvalue and corresponding
  eigenvector of a 2x2 matrix.

  The matrix is stored as 
  arr[2:] = Ixx, Iyy, Ixy, Iyx

  The matrix is supposed to be symmetric, but it is necessary to explicity
  symmetrize due to numerical errors.

  The storage oddness is because of the application.
  """
  cdef double rxx, ryy, rxy, theta, s, c, e1, e2, tmp
  rxx = arr[2]
  ryy = arr[3]
  rxy = (arr[4]+arr[5])/2.0


  if rxx == ryy :
    theta = atan(1.0)
  else :
    tmp = (2.0*rxy)/(ryy-rxx)
    theta = atan(tmp)/2.0
  s = sin(theta)
  c = cos(theta)

  e1 = c*c*rxx +  s*s*ryy - 2*c*s*rxy
  e2 = s*s*rxx + c*c*ryy + 2*c*s*rxy

  cdef np.ndarray[np.float64_t, ndim=1] out
  out = np.zeros(3, dtype='f8')

  if fabs(e1) > fabs(e2) :
    out[0] = e1
    out[1] = c
    out[2] = -s
  else :
    out[0] = e2
    out[1] = s
    out[2] = c

  return out



