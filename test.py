import numpy as np
from scipy.special import sph_harm
from numpy.random import random_sample

import matplotlib.pyplot as plt

import ssrfpy



#generate a bunch of random points on the surface of a sphere
n=10000
x = random_sample(n)-0.5
y = random_sample(n)-0.5
z = random_sample(n)-0.5

for i, (xval,yval,zval) in enumerate(zip(x,y,z)):
    norm = np.sqrt(xval*xval+yval*yval+zval*zval)
    x[i] = x[i]/norm
    y[i] = y[i]/norm
    z[i] = z[i]/norm

colats = np.arccos( z/np.sqrt(x*x+y*y+z*z))
lats = np.pi/2. - colats 
lons = np.arctan2( y, x )+np.pi

#evaluate a spherical harmonic on that sphere
vals = sph_harm(4,9, lons, colats).real
vals = vals/np.amax(vals)


#interpolate onto a regular grid
meshlon, meshlat, values = ssrfpy.interpolate_regular_grid( lons, lats, vals, degrees=False) 

#Plot the result
plt.contourf( meshlon, meshlat, values)
plt.scatter(lons, lats, s=1)
plt.show()

