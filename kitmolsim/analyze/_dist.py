"""
helper module for calculating distances
"""
import os
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

flibname = "libflib.so"
base_dir = os.path.dirname(os.path.abspath(__file__))
flibpath = os.path.join(base_dir, "flib")
fortlib = np.ctypeslib.load_library(flibname, flibpath)
##############################################################################
fortlib.fort_calc_distance_pbc.argtypes = [ctypes.POINTER(ctypes.c_int),
                                    ndpointer(dtype=np.float64),
                                    ndpointer(dtype=np.float64),
                                    ndpointer(dtype=np.float64)]
fortlib.fort_calc_distance_pbc.restype = None


def calc_distance_with_pbc(r, L):
    """
    compute distances betwen pairs of particles in a periodic domain.
    
    Parameters
    -----------
    r  (np.ndarray) : other particle positions. size = (N,3) where N is a number of particles.
    L  (np.ndarray) : edge lengths of a simulation box (x,y,z)

    return: distance matrix between pairs of particles. size = (N,)
    """

    N = len(r)
    mat = np.zeros((N,N))
    fortlib.fort_calc_distance_pbc(ctypes.byref(ctypes.c_int(N)), L, r, mat)
    return mat


##############################################################################
fortlib.fort_calc_distance_p2_pbc.argtypes = [ctypes.POINTER(ctypes.c_int), #N
                                    ndpointer(dtype=np.float64),            #L
                                    ndpointer(dtype=np.float64),            #r1
                                    ndpointer(dtype=np.float64),            #r2
                                    ndpointer(dtype=np.float64)]            #dist
fortlib.fort_calc_distance_p2_pbc.restype = None


def calc_distance_with_p2_pbc(r1, r, L):
    """
    compute distance betwen a particle and other particles in a periodic domain.
    
    Parameters
    -----------
    r1 (np.ndarray) : a particle position. size = (3,)
    r  (np.ndarray) : other particle positions. size = (N,3) where N is a number of particles.
    L  (np.ndarray) : edge lengths of a simulation box (x,y,z)

    return: distances between a particle and others. size = (N,)
    """
    N = len(r)
    dist = np.zeros(N)
    fortlib.fort_calc_distance_p2_pbc(ctypes.byref(ctypes.c_int(N)), L, r1, r, dist)
    
    return dist
