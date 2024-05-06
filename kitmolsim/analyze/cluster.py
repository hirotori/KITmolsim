import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# available if numba exists
try:
    from numba import njit
    print("distance computation: using jit")
    @njit(cache=True)
    def calc_distance_with_pbc(r:np.ndarray, L):
        """Define function computing distance between two points with periodic boundary condition"""
        _n = len(r)
        dr = np.zeros((_n,_n))
        for i in range(_n-1):
            u = r[i,:]
            for j in range(i, _n):
                v = r[j,:]
                _dr = np.abs(u - v)                
                _dr = np.where(_dr > L / 2, _dr-L, _dr)
                _dr = np.sqrt(np.sum(_dr ** 2))
                dr[i,j] = _dr
                dr[j,i] = _dr
        return dr
except:
    import os
    # if you have fortran compiler, you can use fortran library for computing dinstance matrix.
    flibpath = "./lib/"
    flibname = "libf90_distance.so"
    # check if fortran library exists
    if os.path.exists(os.path.join(flibpath, flibname)):
        print("distance computation: using fortran library")
        from numpy.ctypeslib import ndpointer
        import ctypes
        fortlib = np.ctypeslib.load_library(flibname, flibpath)
        fortlib.fort_calc_distance_pbc.argtypes = [ctypes.POINTER(ctypes.c_int),
                                            ndpointer(dtype=np.float64),
                                            ndpointer(dtype=np.float64),
                                            ndpointer(dtype=np.float64)]
        fortlib.fort_calc_distance_pbc.restype = None

        def calc_distance_with_pbc(r, L):
            N = len(r)
            mat = np.zeros((N,N))
            fortlib.fort_calc_distance_pbc(ctypes.byref(ctypes.c_int(N)), L, r, mat)
            return mat
    else:
        print("distance computation: using scipy")
        # without using numba
        def modified_distance(u, v, L):
            """Define function computing distance between two points with periodic boundary condition"""
            diff = np.abs(u - v)
            diff = np.where(diff > L / 2, diff-L, diff)
            return np.sqrt(np.sum(diff ** 2))

        def calc_distance_with_pbc(r:np.ndarray, L):
            return squareform(pdist(r, metric=lambda u, v: modified_distance(u, v, L)))


    
def get_cluster_info(coordinates, R, L, additional_process=None):
    """
    Perform cluster analysis on 3D coordinates using DBSCAN.

    Parameters:
    coordinates (ndarray): Array of 3D unwrapped coordinates.
    R (float): Radius for clustering.
    additional_process: callback function for additional process after clustering. It is called immediately after DBSCAN, taking cluster labels as an argument. 

    Returns:
    Tuple: A tuple containing the number of clusters, sizes of each cluster, and a dictionary with cluster information.

    Example:
    >>> R = 1.0
    >>> num_particles = 10
    >>> coordinates = np.random.rand(num_particles, 2) * 10  # Assuming a 10x10 space
    >>> num_clusters, cluster_sizes, cluster_info = get_cluster_info(coordinates, R)
    >>> print("Number of clusters:", num_clusters)
    >>> print("Cluster sizes:", cluster_sizes)
    >>> print("Cluster information:", cluster_info)
    Number of clusters: 11
    Cluster sizes: [5, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2]
    Cluster information: {0: [0, 5, 7, 13, 19], 1: [1], 2: [2, 8], 3: [3, 9, 10], 4: [4], 5: [6], 6: [11, 15], 7: [12], 8: [14], 9: [16], 10: [17, 18]}
    """
    dist_matrix = calc_distance_with_pbc(coordinates, L)
    clusters = DBSCAN(eps=R, metric="precomputed", min_samples=1).fit_predict(dist_matrix)
    if additional_process is not None:
        clusters = additional_process(clusters)
    # print(clusters, clusters.shape)
    cluster_info = {}
    for i, cluster_label in enumerate(clusters):
        if cluster_label not in cluster_info:
            cluster_info[cluster_label] = []
        cluster_info[cluster_label].append(i)

    num_clusters = len(cluster_info)
    cluster_sizes = np.array([len(cluster) for cluster in cluster_info.values()], dtype=np.int32)
    return num_clusters, cluster_sizes, cluster_info

def calc_mean_aggregation_number(cluster_sizes, NC):
    PM_vs_time = np.zeros(NC)
    sum_sizes = cluster_sizes.sum() # same as `NC`
    M_in_system, num_M = np.unique(cluster_sizes, return_counts=True)
    PM_vs_time[M_in_system] = num_M*M_in_system/NC
    return PM_vs_time
