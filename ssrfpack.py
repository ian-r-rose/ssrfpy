from __future__ import division
from __future__ import print_function

import ctypes
import numpy as np
from scipy.special import sph_harm
from numpy.random import random_sample
import matplotlib.pyplot as plt

ssrfpack = ctypes.CDLL("./libssrfpack.so")

class stripack_triangulation( object ):
    def __init__( self, x, y, z, vals):
        self.n = len(x)
        self.tria_list = np.empty( 6*n, dtype='int64' )
        self.tria_lptr = np.empty( 6*n, dtype='int64' )
        self.tria_lend = np.empty( n, dtype='int64' )
        self.tria_lnew = ctypes.c_int64()
        self.x = x
        self.y = y
        self.z = z
        self.vals = vals

def create_trmesh( x, y, z, vals ):
    #Do some type and length verification
    assert( isinstance( x, np.ndarray ) )
    assert( isinstance( y, np.ndarray ) )
    assert( isinstance( z, np.ndarray ) )
    assert( isinstance( vals, np.ndarray ) )
    assert( x.ndim == 1 )
    assert( y.ndim == 1 )
    assert( z.ndim == 1 )
    assert( vals.ndim == 1 )
    assert( len(x) == len(y) and len(x) == len(z) == len(vals))

    #get the length
    n = ctypes.c_int64( len(x) )

    #allocate space for the triangulation structure
    tria = stripack_triangulation( x, y, z, vals)
    
    #allocate workspace
    work_near = np.empty( n.value, dtype='int64' )
    work_next = np.empty( n.value, dtype='int64' )
    work_dist = np.empty( n.value, dtype='float64' )
 
    #error handling
    ier = ctypes.c_int64()

    #call stripack to create the triangulation
    trmesh_ = ssrfpack.trmesh_
    trmesh_( ctypes.byref(n),\
             tria.x.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
             tria.y.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
             tria.z.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
             tria.tria_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
             tria.tria_lptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
             tria.tria_lend.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
             ctypes.byref( tria.tria_lnew ),\
             work_near.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
             work_next.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
             work_dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
             ctypes.byref( ier ) )
 
    #Check the error code
    if ier.value != 0:
        errors = { -1 : "The number of nodes is less than three.",\
                   -2 : "The first three nodes are colinear."}
        print("Exited with Error: ", ier.value )
        if ier.value < 1:
            print(errors[ier.value])

    #return the triangulation
    return tria

def intrc0( lats, lons, tria, degrees = True ):
    assert( lats.shape == lons.shape )
    shape = lats.shape    

    #Easier to iterate over all the elements when flattened
    flatlats = lats.ravel()
    flatlons = lons.ravel()
    flatvals = np.empty_like( flatlats )

    #Convert to degrees if necessary
    if degrees == True:
        flatlats = np.deg2rad( flatlats )
        flatlons = np.deg2rad( flatlons )
    
    #number of nodes
    n = ctypes.c_int64( tria.n )
    #error handling
    ier = ctypes.c_int64()
    #cached node index
    ist = ctypes.c_int64(1)

    #loop over the requested lat-lons and interpolate based on the 
    #triangulation.
    for i, (lat, lon) in enumerate(zip(flatlats, flatlons)):
        val = ctypes.c_double()
        plat = ctypes.c_double( lat ) # wrap the lat
        plon = ctypes.c_double( lon ) # wrap the lon
        
        intrc0_ = ssrfpack.intrc0_
        intrc0_( ctypes.byref(n),\
                 ctypes.byref(plat),\
                 ctypes.byref(plon),\
                 tria.x.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
                 tria.y.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
                 tria.z.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
                 tria.vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double) ),\
                 tria.tria_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
                 tria.tria_lptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
                 tria.tria_lend.ctypes.data_as(ctypes.POINTER(ctypes.c_int64) ),\
                 ctypes.byref( ist ),\
                 ctypes.byref( val ),\
                 ctypes.byref( ier ) )
                 
        flatvals[i] = val.value

    values = flatvals.reshape( shape )
    return values
        
        
n=10000
x = random_sample(n)-0.5
y = random_sample(n)-0.5
z = random_sample(n)-0.5

for i, (xval,yval,zval) in enumerate(zip(x,y,z)):
    norm = np.sqrt(xval*xval+yval*yval+zval*zval)
    x[i] = x[i]/norm
    y[i] = y[i]/norm
    z[i] = z[i]/norm

lats = np.arccos( z/np.sqrt(x*x+y*y+z*z))
lons = np.arctan2( y, x )+np.pi

#evaluate a spherical harmonic on that sphere
vals = sph_harm(4,9, lons, lats).real
vals = vals/np.amax(vals)

tria = create_trmesh(x,y,z,vals)
reglat = np.linspace(-90.,90., 91)
reglon = np.linspace(0.,360., 181)
meshlat,meshlon = np.meshgrid(reglat,reglon)

values = intrc0( meshlat, meshlon, tria)

plt.contourf( meshlon, meshlat, values)
plt.show()

