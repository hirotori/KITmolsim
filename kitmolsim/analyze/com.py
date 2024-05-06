"""
Calculate center of mass of cluster under periodic boundary condition.

Main idea is refered from:

L. Bai and D. Breen, Journal of Graphics Tools, 2008, 13, 53-60.

Slightly modified by Torsten2001's answer in stack overflow:

https://stackoverflow.com/questions/18166507/using-fft-to-find-the-center-of-mass-under-periodic-boundary-conditions 

"""

import numpy as np

def calc_cluster_com(points:np.ndarray, box_size:float, shift_points=False) -> np.ndarray:
    """
    calculate center of mass. ll points are supposed to be in a rectangle box.
    
    Parameter
    ----------

    points : (N,3) ndarray
        point coordinates of a cluster

    box_size : float
        simulation box dimension. all lengths are supposed to be same.

    shift_points : bool
        set True if the simulation box is centered at the origin of coordinates.
    """

    # map points to -pi ... pi
    ri = box_size/(2.0*np.pi)
    if shift_points:
        points_map = (points + box_size/2)/ri - np.pi
    else:
        points_map = points/ri - np.pi

    pdim = points.shape[1]
    cluster_com = np.empty(pdim)
    for d in range(pdim):
        cluster_com[d] = ri*(np.arctan2(np.sin(points_map[:,d]).mean(), np.cos(points_map[:,d]).mean()) + np.pi)

    if shift_points:
        cluster_com -= box_size/2

    return cluster_com


