# ssrfpy
This is a small Python project for interfacing the Fortran codes STRIPACK and SSRFPACK to the larger Python
ecosystm. These libraries were written by Robert Renka and published by ACM.

STRIPACK is a set of related functions for Delaunay and Voronoi mesh generation on the surface of a sphere.

SSRFPACK uses the triangulations generated by STRIPACK to perform interpolation of an arbitrary function on a sphere.

I have wrapped the most basic functionality of these packages for triangulation and interpolation.
I may include more as the need arises, but it currently only performs linear interpolation and cubic interpolation onto regular grids.

#Installation

This package has numpy as dependency, and requires a C compiler like gcc.  
You can install it by going to the root directory of the project and typing

```python setup.py install```

You may want to add the ```--prefix=/path/to/intall/directory``` option, depending upon where your Python distribution lives.

Once it is installed, you can test it by entering the ```examples``` directory and running one of the example scripts.
