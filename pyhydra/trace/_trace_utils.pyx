import numpy as np
cimport numpy as np
cimport cython

# Transform the derivative array into a line-point array
# Reuse the data as follows 
#   0 -- max eigenvalue
#   1 -- angle
#   2 -- px 
#   3 -- py
#   4 -- flag : 0 = not a line point  
#               1 = line point for start
#               2 = line start for line
#   5 -- flag : 0 = not assigned
#               1 = assigned

# Get the C-routines for arithmetic
cdef extern from "math.h" :
  double atan(double)
  double atan2(double, double)
  double sin(double)
  double cos(double)
  double fabs(double)
  double sqrt(double)
  double exp(double)


# Turn off error checking!!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] _gauss(np.ndarray[np.float64_t, ndim=1] xc, 
                                              np.ndarray[np.float64_t, ndim=1] yc, double sigma):
  cdef np.ndarray[np.float64_t, ndim=2] kern
  cdef long nx, ny, ix, iy
  cdef double xx, yy, tmp, den, norm

  nx = xc.size
  ny = yc.size

  den = 2.0*sigma*sigma
  norm = 1./(2.0*np.pi*sigma**2)

  kern = np.zeros((nx, ny), dtype='f8')

  for ix in range(nx):
    xx = xc[ix]
    if (xx/sigma) > 5 :
      continue
    for iy in range(ny):
      yy = yc[iy]
      tmp = (xx*xx + yy*yy)/den
      if tmp < 20.0 :
        kern[ix, iy] = norm*exp(-tmp)
  return kern

# Turn off error checking!!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] _gauss_x(np.ndarray[np.float64_t, ndim=1] xc, 
                                               np.ndarray[np.float64_t, ndim=1] yc, double sigma):
  cdef np.ndarray[np.float64_t, ndim=2] kern 
  cdef long nx, ny, ix, iy
  cdef double xx, yy, tmp, den, norm

  nx = xc.size
  ny = yc.size

  den = 2.0*sigma*sigma
  norm = 1./(2.0*np.pi*sigma**4)

  kern = np.zeros((nx, ny), dtype='f8')

  for ix in range(nx):
    xx = xc[ix]
    if (xx/sigma) > 5 :
      continue
    for iy in range(ny):
      yy = yc[iy]
      tmp = (xx*xx + yy*yy)/den
      if tmp < 20.0 :
        kern[ix, iy] = -norm*xx*exp(-tmp)
  return kern

# Turn off error checking!!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] _gauss_y(np.ndarray[np.float64_t, ndim=1] xc, 
                                               np.ndarray[np.float64_t, ndim=1] yc, double sigma):
  cdef np.ndarray[np.float64_t, ndim=2] kern 
  cdef long nx, ny, ix, iy
  cdef double xx, yy, tmp, den, norm

  nx = xc.size
  ny = yc.size

  den = 2.0*sigma*sigma
  norm = 1./(2.0*np.pi*sigma**4)

  kern = np.zeros((nx, ny), dtype='f8')

  for ix in range(nx):
    xx = xc[ix]
    if (xx/sigma) > 5 :
      continue
    for iy in range(ny):
      yy = yc[iy]
      tmp = (xx*xx + yy*yy)/den
      if tmp < 20.0 :
        kern[ix, iy] = -norm*yy*exp(-tmp)
  return kern

# Turn off error checking!!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double jacobi2x2(np.ndarray[np.float64_t, ndim=1] arr):
  """ Compute the maximum eigenvalue and corresponding
  eigenvector of a 2x2 matrix.

  The matrix is stored as 
  arr[2:] = Ixx, Iyy, Ixy, Iyx

  The matrix is supposed to be symmetric, but it is necessary to explicity
  symmetrize due to numerical errors.

  The storage oddness is because of the application.
  """
  cdef double rxx, ryy, rxy, theta, s, c, e1, e2, tmp, emax, nx, ny, t
  cdef double rx, ry
  rx = arr[0]
  ry = arr[1]
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

  if fabs(e1) > fabs(e2) :
    emax = e1
    nx = c
    ny = -s
  else :
    emax = e2
    nx = s
    ny = c

  t = -(rx*nx + ry*ny)/(rxx*nx*nx + 2*rxy*nx*ny + ryy*ny*ny)

  # Refill the Hessian array
  arr[0] = emax
  arr[1] = fabs(atan2(ny, nx)) # Directions are degenerate -- so just use absolute value
  arr[2] = t*nx
  arr[3] = t*ny
  arr[4] = 0.0
  arr[5] = 0.0

  return emax

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double _do_jacobi2x2_all(np.ndarray[np.float64_t, ndim=3] hess):
  cdef long nx, ny, ix, iy
  cdef double rmin, emin
  rmin = 0.0

  # Get the size of the array
  nx = hess.shape[0]-2
  ny = hess.shape[1]-2

  # Loop over all the points
  for ix in range(1, nx+1):
    for iy in range(1, ny+1):
      emin = jacobi2x2(hess[ix,iy,:])
      if rmin > emin :
        rmin = emin

  return rmin


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _saliency(np.ndarray[np.float64_t, ndim=3] hess, double upper, double lower, double eps, long pad):
  cdef long nx, ny, ix, iy

  # Get the size of the array
  nx = hess.shape[0]-2
  ny = hess.shape[1]-2

  # Loop over all the points again 
  for ix in range(1+pad, nx+1-pad):
    for iy in range(1+pad, ny+1-pad):
      if hess[ix, iy, 0] < lower :
        if fabs(hess[ix, iy, 2]) < (0.5 + eps) :
          if fabs(hess[ix, iy, 3]) < (0.5 + eps) :
            if hess[ix, iy, 0] < upper :
              hess[ix,iy,4] = 2
            else :
              hess[ix,iy,4] = 1




# Remove coincident points. This may end up with semi-subtle race conditions 
# but we aren't going to solve that now!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _remove_coincident(np.ndarray[np.float64_t, ndim=3] hess, double eps):
  cdef long nx, ny, ix, iy, ix1, iy1
  cdef double x1, y1, dx, dy

  # Get the size of the array
  nx = hess.shape[0]-2
  ny = hess.shape[1]-2

  # Loop over all the points again 
  for ix in range(1, nx+1):
    for iy in range(1, ny+1):
      # If this is not a salient point, keep going
      if hess[ix, iy, 4] == 0 :
        continue

      # Compute subgrid position of this point
      x0 = hess[ix,iy,2]
      y0 = hess[ix,iy,3]

      # Now loop over all the neighbours
      for ix1 in range(-1, 2):
        for iy1 in range(-1, 2):
          # Ignore myself
          if (ix1 == 0) and (iy1 == 0):
            continue

          # Ignore if not a salient point
          if hess[ix+ix1, iy+iy1, 4] == 0:
            continue

          #Compute subgrid position of this point
          x1 = ix1 + hess[ix+ix1, iy+iy1, 2]
          y1 = iy1 + hess[ix+ix1, iy+iy1, 3]
          # Differences
          dx = fabs(x1-x0)
          dy = fabs(y1-y0)

          # If these points are truly coincident
          # and the current point hasn't been reset
          if (dx < eps) and (dy < eps) and hess[ix,iy,4] > 0:
            if hess[ix, iy, 0] <= hess[ix+ix1, iy+iy1, 0]:
              hess[ix+ix1, iy+iy1, 4] = 0
            else :
              hess[ix,iy,4] = 0



# Compute the difference in angle -- return a value between 0, pi/2
# assumes that the two angles are positive and that they are between 0, pi
cdef double _diff_angle(double a1, double a2):
  cdef double pi = 1.5707963267948966
  cdef double diff

  diff = fabs(a1 - a2)
  if diff > (pi/2) :
    diff = pi - diff
  return diff

# This uses the default d + c\beta metric for choosing points
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _grow_line(np.ndarray[np.float64_t, ndim=3] hess, int ix0, int iy0):
  cdef double x0, y0, dx, dy, beta, r, alpha1, alpha2
  cdef int ix, iy, ii, jj
  cdef int imin, jmin
  
  donext = True

  # Starting point
  ix = ix0
  iy = iy0
  alpha1 = hess[ix,iy,1]
  x0 = ix0 + hess[ix, iy, 2]
  y0 = iy0 + hess[ix, iy, 3]

  # Start the line list
  line_f = [(x0-1, y0-1)] # We padded to avoid the need to deal with edges

  # Nark the first point as assigned
  hess[ix, iy, 4] = -1

  # Loop until we have no more new points
  while donext :
    # Define rmin
    rmin = None 

    # Now loop over all neighbours
    for ii in range(-1, 2):
      for jj in range(-1, 2):
        # Ignore self point
        if (ii==0) and (jj == 0) :
          continue
        if hess[ix+ii, iy+jj, 4] > 0 :  # Is this a line point
          # Compute the distance
          dx = hess[ix+ii, iy+jj, 2] + ix + ii - x0
          dy = hess[ix+ii, iy+jj, 3] + iy + jj - y0
          alpha2 = hess[ix+ii, iy+jj, 1]
          r = sqrt(dx*dx+dy*dy) + _diff_angle(alpha1, alpha2)

          if (rmin is None) or (r < rmin):
            rmin = r
            imin = ii
            jmin = jj

    # If rmin doesn't get set, close down 
    if rmin is None :
      donext = False
    else :
      ix += imin
      iy += jmin
      alpha1 = hess[ix, iy, 1]
      x0 = ix + hess[ix, iy, 2]
      y0 = iy + hess[ix, iy, 3]
      hess[ix, iy, 4] = -2
      line_f.append((x0-1, y0-1))


  # Set up for the reverse line
  line_r = []
  donext = True
  ix = ix0
  iy = iy0
  alpha1 = hess[ix,iy,1]
  x0 = ix0 + hess[ix, iy, 2]
  y0 = iy0 + hess[ix, iy, 3]
  while donext :
    # Define rmin
    rmin = None 

    # Now loop over all neighbours
    for ii in range(-1, 2):
      for jj in range(-1, 2):
        # Ignore self point
        if (ii==0) and (jj == 0) :
          continue
        if hess[ix+ii, iy+jj, 4] > 0 :  # Is this a line point
          # Compute the distance
          dx = hess[ix+ii, iy+jj, 2] + ix + ii - x0
          dy = hess[ix+ii, iy+jj, 3] + iy + jj - y0
          alpha2 = hess[ix+ii, iy+jj, 1]
          r = sqrt(dx*dx+dy*dy) + _diff_angle(alpha1, alpha2)

          if (rmin is None) or (r < rmin):
            rmin = r
            imin = ii
            jmin = jj

    # If rmin doesn't get set, close down 
    if rmin is None :
      donext = False
    else :
      ix += imin
      iy += jmin
      alpha1 = hess[ix,iy,1]
      x0 = ix + hess[ix, iy, 2]
      y0 = iy + hess[ix, iy, 3]
      hess[ix, iy, 4] = -2
      line_r.append((x0-1, y0-1))



  # Return
  return (line_f, line_r)
