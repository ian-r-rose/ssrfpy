metadata = dict( name= 'ssrfpy',
                 version = 0.1,
                 description='Functions for interpolating unstructured spherical data',
                 url='',
                 author='Ian Rose',
                 author_email='ian.rose@berkeley.edu',
                 license='GPL',
                 long_description='',
                 packages = ['ssrfpy'],
               )

#Try to use setuptools in order to check dependencies.
#if the system does not have setuptools, fall back on
#distutils.
try:
  from setuptools import setup, find_packages, Extension
  metadata['install_requires'] = ['numpy', 'matplotlib']
except ImportError:
  from distutils.core import setup, Extension

libssrfpack = Extension('ssrfpy._ssrfpack', \
                        sources=['ssrfpy/ssrfpack.c', 'ssrfpy/stripack.c'],\
                        libraries=['m'])
metadata['ext_modules'] = [libssrfpack]


setup ( **metadata )
