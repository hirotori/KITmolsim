import numpy as np


def compute_inertia_tensor(r:np.ndarray, m:np.ndarray, com=True):
    """
    compute inertia tensor of mass particle system.

    Parameter
    -----------
    r (np.ndarray) : position of mass particles, shape = (N, 3)
    m (np.ndarray) : mass of particles, shape = (N, )
    com (bool) : the tensor is calculated in a reference frame 
    """

    npoint, ndim = r.shape
    if ndim != 3: raise ValueError("the 2nd dim of 'r' must be 3.")
    if npoint != len(m): raise ValueError("the length of 'm' not consistent with the 1st dim of 'r'")

    if com:
        r_com = r.mean(axis=0)
        r_work = r - r_com
    else:
        r_work = r

    x, y, z = r_work[:, 0], r_work[:, 1], r_work[:, 2]
    
    # 対角成分の計算
    Ixx = np.sum(m * (y**2 + z**2))
    Iyy = np.sum(m * (x**2 + z**2))
    Izz = np.sum(m * (x**2 + y**2))
    
    # 非対角成分の計算
    Ixy = np.sum(-m * x * y)
    Ixz = np.sum(-m * x * z)
    Iyz = np.sum(-m * y * z)
    
    # 慣性モーメントテンソルの構築
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])    
    
    return I