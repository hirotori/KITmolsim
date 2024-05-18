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


def fcc_unit_cell():
    return np.array(
        [[0.0, 0.0, 0.0], # 0
         [0.0, 0.5, 0.5], # 1
         [0.5, 0.0, 0.5], # 2
         [0.5, 0.5, 0.0], # 3
         ])

def diamond_unit_cell():
    return np.array(
        [[0.0, 0.0, 0.0], # 0
         [0.0, 0.5, 0.5], # 1
         [0.5, 0.0, 0.5], # 2
         [0.5, 0.5, 0.0], # 3
         [0.25, 0.25, 0.25], # 4
         [0.25, 0.75, 0.75], # 5
         [0.75, 0.25, 0.75], # 6
         [0.75, 0.75, 0.25] # 7
        ])

def make_lattice(unit_lattice:np.ndarray, num_cell:Tuple[int], lattice_constant:float):
    """
    create cubic lattice from unit lattice.

    Parameter
    ---------
    unit_lattice (`np.ndarray`) : unit lattice. shape = (N,3) where N = number of atom. all atoms must be in a box of unit length.
    num_cell (tuple(int)) : number of cells along each direction (x,y and z).
    lattice_constant (float) : lattice constant.

    return: atom position (ndarray) and simulation box lengths (Lx,Ly,Lz)
    """
    n_atom_lattice = len(unit_lattice)
    pos = []
    for ix in range(num_cell[0]):
        for iy in range(num_cell[1]):
            for iz in range(num_cell[2]):
                for j in range(0, n_atom_lattice):
                    x = (unit_lattice[j][0]+ix)*lattice_constant - 0.5*num_cell[0]*lattice_constant
                    y = (unit_lattice[j][1]+iy)*lattice_constant - 0.5*num_cell[1]*lattice_constant
                    z = (unit_lattice[j][2]+iz)*lattice_constant - 0.5*num_cell[2]*lattice_constant
                    pos.append([x,y,z])
    
    return np.array(pos), np.array(num_cell)*lattice_constant


def make_defected_fcc(L:float, rho:float, seed:int):
    """
    create defected FCC lattice with given density and box size.
    
    See: https://zenn.dev/kaityo256/articles/md_initial_condition

    """
    # compute num of cells
    m = int(np.floor((L**3 * rho / 4.0)**(1.0 / 3.0)))
    drho1 = np.abs(4.0 * m**3 / L**3 - rho)
    drho2 = np.abs(4.0 * (m + 1)**3 / L**3 - rho)
    
    ncell = m if drho1 < drho2 else m+1

    pos, Lbox = make_lattice(fcc_unit_cell(), (ncell,ncell,ncell), L/m)

    # sampling
    n = int(rho*L**3)
    rng = np.random.Generator(bit_generator=np.random.MT19937(seed=seed))
    pos_defected = rng.choice(pos, size=n, replace=False)

    return pos_defected, Lbox

