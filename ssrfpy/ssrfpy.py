# Copyright (C) 2015, Rose, I
# Released under GPL v3 or later.

from __future__ import division
from __future__ import print_function

import os
import ctypes
import numpy as np

#Try to import jit, but if it is not available
#then just define an identity decorator
try:
    from numba import jit,njit
except ImportError:
    def jit(fn):
        return fn
    njit = jit

PATH=os.path.dirname(__file__)
ssrfpack = ctypes.CDLL(PATH+"/_ssrfpack.so")

class _stripack_triangulation( object ):
    """
    Class wrapping the arrays that STRIPACK uses 
    to construct a triangulation over a sphere.
    """
    def __init__( self, x, y, z, vals):
        self.n = len(x)
        self.tria_list = np.empty( 6*self.n, dtype='int64' )
        self.tria_lptr = np.empty( 6*self.n, dtype='int64' )
        self.tria_lend = np.empty( self.n, dtype='int64' )
        self.tria_lnew = ctypes.c_int64()
        self.x = x
        self.y = y
        self.z = z
        self.vals = vals

def _create_triangulation( x, y, z, vals ):
    """
    Given Cartesian points on the surface of a sphere 
    and a vector of function values at those points, 
    construct the Delaunay triangulation for those points.
    The function values are not used in this, except to 
    be stored in the triangulation object that is returned.
    """

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
    tria = _stripack_triangulation( x, y, z, vals)
    
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
        else:
            print("There were duplicate vertices, which cannot be triangulated.")

    #return the triangulation
    return tria

def _linear_interpolate( lons, lats, tria, degrees = True ):
    """
    Interpolate a function onto (lons, lats). The longitudes 
    and latitudes may be arbitrary, and the triangulation is 
    created with _create_triangulation().  If lons and lats 
    are in degrees, set degrees=True.  If they are radians, 
    set degrees=False
    """
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

def lon_lat_to_cartesian( lons, lats , degrees = True):
    """
    Small utility function that takes longitudes and latitudes
    and converts them the Cartesian coodinates on a unit sphere.
    Set degrees=True if they are measured in degrees, and False
    if they are measured in radians.
    """
    x = np.empty_like(lons)
    y = np.empty_like(lons)
    z = np.empty_like(lons)
    if degrees:
        x = np.cos( np.deg2rad(lons) )*np.cos(np.deg2rad(lats) )
        y = np.sin( np.deg2rad(lons) )*np.cos(np.deg2rad(lats) )
        z = np.sin( np.deg2rad(lats) )
    else:
        x = np.cos(lons)*np.cos(lats)
        y = np.sin(lons)*np.cos(lats)
        z = np.sin(lats)
    return x,y,z

#Unclear whether jitting this function is worth it
@jit
def remove_duplicates( lons, lats, vals ):
    """
    Given the lons, lats, vals tuple, remove duplicated entries,
    since Delaunay triangulation has a hard time with those.
    It's unclear if this is a sensible implementation, though it 
    seems to work. I sort the lon,lat,val tuples according to 
    lon and lat, and if successive entries are equal, then 
    don't include one in the final list. This is a looping heavy
    function, so we try to jit it with numba, if available.
    """
    cleaned_lons = np.empty_like(lons)
    cleaned_lats = np.empty_like(lats)
    cleaned_vals = np.empty_like(vals)

    n = len(lons)
    indices = np.lexsort( (lons, lats ) )

    #First point is always going to be unique
    cleaned_lons[0] = lons[indices[0]]
    cleaned_lats[0] = lats[indices[0]]
    cleaned_vals[0] = vals[indices[0]]

    j = 0 # counter for cleaned_lats,cleaned_lons
    i = 1 # counter for lat,lons
    while i < n:
        if (lats[indices[i]] == cleaned_lats[j]) and (lons[indices[i]]==cleaned_lons[j]):
            i += 1
        else:
            j += 1
            cleaned_lons[j] = lons[indices[i]]
            cleaned_lats[j] = lats[indices[i]]
            cleaned_vals[j] = vals[indices[i]]
            i += 1
    cleaned_lons = np.delete( cleaned_lons, slice(j, None, None))
    cleaned_lats = np.delete( cleaned_lats, slice(j, None, None))
    cleaned_vals = np.delete( cleaned_vals, slice(j, None, None))
    return cleaned_lons, cleaned_lats, cleaned_vals
        
def interpolate_regular_grid( lons, lats, values, n=90, degrees=True, use_legendre=False ):
    """
    Given one dimensional arrays for longitude, latitude, and a function
    evaluated at those points, construct a regular grid and interpolate the
    function onto that grid.  The parameter ``n'' sets the resolution of the grid, 
    where there are n+1 points in latitude and 2n+1 points in longitude.
    The degrees parameter should be True if longitude and latitude are 
    measured in degrees, False if they are measured in radians.
    """
    cleaned_lons, cleaned_lats, cleaned_values = remove_duplicates( lons, lats, values )
    x,y,z = lon_lat_to_cartesian( cleaned_lons, cleaned_lats , degrees)

    #Figure out the resolution
    nlat = 0
    nlon = 0
    if isinstance( n, tuple):
        assert( len(n) == 2 )
        nlon = 2*n(0)+1
        nlat = n(1)+1
    else:
        nlon = 2*n+1
        nlat = n+1

    reg_lat = 0.0
    if use_legendre:
        raise Exception("Legendre points in latitude not yet implemented")
    else:
        reg_lat = np.linspace(-np.pi/2., np.pi/2., nlat, endpoint=True)
    reg_lon = np.linspace(0., 2.*np.pi, nlon, endpoint = True )
    if (degrees):
        reg_lon = np.rad2deg(reg_lon)
        reg_lat = np.rad2deg(reg_lat)
    mesh_lon, mesh_lat = np.meshgrid(reg_lon, reg_lat)

    tria = _create_triangulation(x, y, z, cleaned_values)
    mesh_values = _linear_interpolate( mesh_lon, mesh_lat, tria, degrees=degrees)

    return mesh_lon, mesh_lat, mesh_values
