import numpy as np

def calc_gyration_tensor(r:np.ndarray) -> np.ndarray:
    """
    calculate gyration tensor of point clouds.
    """
    return np.einsum("im,in -> mn", r, r)/len(r)

def calc_gyration_of_cube(cluster_points:np.ndarray, com:np.ndarray, Nv:float, box:np.ndarray):
    """
    calculate gyration tensor of cluster composed of cubic nanoparticles.
    First, all beads are translated to origin, and then the gyration tensor is calculated based on the center of mass of nanoparticles.

    Parameters
    ------------
    cluster_points : (Ncls*Nv,3)ndarray
        coordinates of beads of nanoparticles composing of a cluster.
        These points must be **wrapped** in periodic boundary box
        because they are unwrapped in this function.
    com : (3,)ndarray
        center of mass of cluster.
        It must be calculated in advance by `com_pb.calc_clsuter_com`.
    Nv : float
        The number of beads which consists of a nanoparticles.
    box : (3,)ndarray
        Simulation box.
    
    Note
    ------------
    If cluster is composed of 1 nanoparticle (i.e. single nanoparticle), then it returns a zero vector and
    unit tensor.

    """
    # translate to origin (almost all of the cluster will be in the box, except large one (M>20))
    dL = box/2
    map_to_center = (cluster_points - com + dL)%box - dL

    # create centers of mass of nano-particles
    com_nps = map_to_center.reshape([-1,Nv,3]).mean(axis=1)

    # calculate gyration tensor
    com = com_nps.mean(axis=0)

    return calc_gyration_tensor(com_nps - com)

def shape_descriptors(gyr_tensor:np.ndarray):
        """
        calculate shape descriptors from gyration tensor.

        Parameter
        ------------
        gyr_tensor: (3,3)ndarray
            gyration tensor.

        Returns
        -------------
        eigval: (3,)ndarray
            eigen values of gyration tensor. sorted in ascending order.
        eigvec: (3,3)ndarray
            eigen vectors of gyration tensor.
            eigvec[:,i] is the corresponding vector with the eigen value eigval[i].
        rg_sq: float
            Gyration radius.
        aspher: float
            asphericity
        ascyli: float
            ascylindricity
        shape_aniso: float
            shape factor. if rg_sq == 0, then shape factor = 0. 

        """
        eigval, eigvec = np.linalg.eigh(gyr_tensor) #instead of eig (complex values are sometimes obtained using eig.)
        sort_indx = np.argsort(eigval)    
        eigval = eigval[sort_indx]
        eigvec = eigvec[:,sort_indx]

        rg_sq   = np.sum(eigval)
        aspher = 1.5*eigval[2] - 0.5*rg_sq
        ascyli = eigval[1] - eigval[0]
        shape_aniso = (aspher*aspher + 0.75*ascyli*ascyli)/(rg_sq*rg_sq) if rg_sq != 0 else 0# Wikipedia. 

        return eigval, eigvec, rg_sq, aspher, ascyli, shape_aniso