import numpy as np
from typing import Tuple

def randomize_positions(Ncol:float, xrange, yrange, zrange, points:np.ndarray, pairs:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    set colloids in the box randomly. 

    Parameters
    -----------
    Ncol : int
        number of colloids
    xrange, yrange, zrange : ndarray
        range of location
    spacing : float

    points : ndarray
        point coordinates of a colloid
    pairs : ndarray
        pair Ids of a colloid

    Returns
    --------
    all_points, all_pairs : ndarray
        points and pairs of all colloids.

    """
    _x, _y, _z = np.meshgrid(xrange, yrange, zrange, indexing="xy")
    r_popu = np.column_stack((_x.ravel(), _y.ravel(), _z.ravel()))
    Ncolloids_max = len(r_popu)
    if Ncol >= Ncolloids_max:
        raise ValueError(f"Given number of colloids greater than maximum size: {Ncol} >= {Ncolloids_max}")
    genrand = np.random.Generator(np.random.MT19937())
    center_positions = genrand.choice(r_popu, size=Ncol, replace=False)

    all_points = np.concatenate(tuple((points.copy()+center_positions[i]) for i in range(Ncol)))
    # 0        ~  Nverts         :: 1st particle (including central point)
    # Nverts+1 ~  2(Nverts) + 1  :: 2nd particle ( 〃 )
    # :         :                :: :
    Ntot = len(pairs)
    Nverts = len(points)

    try:
        all_pairs = np.concatenate(tuple(pairs.copy()+n*Nverts for n in range(Ncol)), dtype=np.int32)
    except TypeError: # for ver 1.19.5
        all_pairs = np.concatenate(tuple(pairs.copy()+n*Nverts for n in range(Ncol))).astype(np.int32)

    return all_points, all_pairs
