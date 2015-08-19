from __future__ import print_function
from __future__ import division

import numpy as np
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

#Calculate latitude and longitude of the random points
colats = np.arccos( z/np.sqrt(x*x+y*y+z*z))
lats = np.pi/2. - colats 
lons = np.arctan2( y, x )+np.pi

#evaluate a Fisher distribution on the surface of a sphere
xm = 1./np.sqrt(3.)
ym = 1./np.sqrt(3.)
zm = 1./np.sqrt(3.)
vals = np.empty(n)
kappa = 1.0 #concentration parameter
c = kappa / (4.*np.pi * np.sinh(kappa) )# normalization constant
for i in range(n):
    vals[i] = c * np.exp( kappa * (x[i]*xm + y[i]*ym + z[i]*zm) )

#interpolate onto a regular grid with C0 interpolation
meshlon, meshlat, values, weights = ssrfpy.interpolate_regular_grid( lons, lats, vals, method = 'linear', degrees=False, use_legendre=True) 

#Compute the integral
tot = np.sum( values * weights ) 
err = np.abs((1.0-tot)/tot * 100.)
print("Integral of the Fisher distribution: ", tot, ", or %5.2e"%(err), "percent error") 

#Plot the result
plt.contourf( meshlon, meshlat, values)
plt.scatter(lons, lats, s=1)
plt.show()
