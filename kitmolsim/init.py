import numpy as np
from typing import Tuple, Union

def create_random_state(N:int, Lbox, kT:float, seed:int, overlap=True, diameter=None):
    """
    create a state of randomized positions and velocity.

    Positions are randomized by sampling points from uniform distribution.
    All particles are placed in the rectangle box centered at (0,0,0). 
    Velocity are randomized by sampling points from Gaussian distribution.

    Parameters
    ----------
    N (int) : number of particles
    Lbox (tuple, list or np.ndarray of size 3) : box dimension
    kT (float) : system temperature
    seed (int) : random seed
    overlap (bool) : whether accept overlapping atoms or not. True by default.
    diameter (float) : diameter of particle
    """

    rng = np.random.Generator(np.random.MT19937(seed=seed))
    if overlap:
        pos = rng.uniform(size=(N,3))
        pos[:,0] = pos[:,0]*Lbox[0] - Lbox[0]/2
        pos[:,1] = pos[:,1]*Lbox[1] - Lbox[1]/2
        pos[:,2] = pos[:,2]*Lbox[2] - Lbox[2]/2
    else:
        if diameter is None:
            raise ValueError("dimaeter must be float.")
        pos = placing_particles_without_overlapping(N, Lbox, rng, diameter=diameter)
    
    vel = rng.normal(loc=0.0, scale=np.sqrt(kT), size=(N,3))

    return pos, vel


def placing_particles_without_overlapping(N:int, Lbox, rng:np.random.Generator, diameter:Union[float,np.ndarray],
                                           obstacle_coms:np.ndarray=None, obstacle_diameter:Union[float,np.ndarray]=None):
    """
    placing particles without overlapping. 

    Parameters
    ----------
    N (int) : number of particles
    rng : random generator
    diameter (float or array like) : diameter of particles. Any particle cannot exist at a distance less than this from any other particle.
    obstacle_coms (ndarray,optional) : center-of-mass of obstacles. particles are placed without overlapping these obstacles.
    obstacle_diameter (float or array like,optional) : diameter of obstacles. 
    """
    pos = []
    iL = 1/Lbox
    obst_exists = obstacle_coms is not None and obstacle_diameter is not None
    # first particle
    ncount = 0
    while ncount != 1:
        ri = rng.random(size=3)
        ri[0] = ri[0]*Lbox[0] - Lbox[0]/2
        ri[1] = ri[1]*Lbox[1] - Lbox[1]/2
        ri[2] = ri[2]*Lbox[2] - Lbox[2]/2
        pos.append(ri)
        if obst_exists:
            if __can_insert_particle(ri, obstacle_coms, Lbox, diameter, obstacle_diameter):
                ncount += 1
            else:
                del pos[-1]
        else:
            # any particle is accepted.
            ncount += 1
    assert(len(pos) == 1)

    while ncount < N:
        print(f"\r kitmolsim::init::_placing: n = {ncount}", end="")
        # draw particle
        ri = rng.random(size=3)
        ri[0] = ri[0]*Lbox[0] - Lbox[0]/2
        ri[1] = ri[1]*Lbox[1] - Lbox[1]/2
        ri[2] = ri[2]*Lbox[2] - Lbox[2]/2

        # Test the overlap between the newly added particle and the other particles.
        dij = __calc_distance_with_pbc(ri, pos, Lbox)
        if all(dij >= diameter):
            # test the overlap between new particle and the obstacles
            if obst_exists:
                if __can_insert_particle(ri, obstacle_coms, Lbox, diameter, obstacle_diameter):
                    pos.append(ri)
                    ncount += 1
                # nothing done if the vector ri is not accepted.
            # if no obstacles exist
            else:
                pos.append(ri)
                ncount += 1


    print("")
    return np.array(pos)

def __can_insert_particle(ri, obst_coms, Lbox, d, obs_d):
    dij_obs = __calc_distance_with_pbc(ri, obst_coms, Lbox)
    if all(dij_obs >= 0.5*(d+obs_d)):
        return True
    else:
        return False

def __calc_distance_with_pbc(ri, pos_others, Lbox):
    diff = np.abs(pos_others - ri)
    diff = np.where(diff > Lbox / 2, diff-Lbox, diff)
    dij = np.sqrt(np.sum(diff*diff, axis=1))

    return dij
    
def placing_particles_fcc(rho:float, Ntarget:int, Lbox, rng:np.random.Generator, 
                          obstacle_coms:np.ndarray=None, obstacle_diameter:float=None):
    """
    Placing atoms in the box on the FCC structure. 
    Particles are first inserted on the FCC structure at a density rho* > `rho`, then adjusting the number of particles
    to reach the target density `rho` by sampling points. 
    If there are obstacles in the box, particles inside the obstacles are removed. 
    If resulting number of particles does not match `Ntarget`, then particles are removed or added to match `Ntarget`.

    Parameters
    -------------
    rho (float) : number density of a system.
    Ntarget (int) : target number of particles.
    Lbox (tuple or array like) : system box
    rng : random generator
    obstacle_coms (ndarray,optional) : center-of-mass of obstacles. particles are placed without overlapping these obstacles.
    obstacle_diameter (float,optional) : diameter of obstacles. 

    Note
    --------------
    Generally, inserting many additional particles takes much time than erasing excess ones.
    We recommend setting `rho` slightly larger than the target number density.

    """

    r_water, _ = make_defected_fcc(L=Lbox[0], rho=rho, seed=442)
    print(f"First generating {len(r_water)} beads")

    # delete atoms
    r_water_new = []
    for rw in r_water:
        dij_obs = __calc_distance_with_pbc(rw, obstacle_coms, Lbox)
        if np.all(dij_obs >= obstacle_diameter):
            r_water_new.append(rw)
    print(f"Water beads deleted: {len(r_water)} ==> {len(r_water_new)}")
    r_water = np.array(r_water_new)

    Nwater_inserted = r_water.shape[0]
    nwater_rest = Ntarget - Nwater_inserted
    if nwater_rest > 0:
        # this process is slow. 
        _obst_coms = np.concatenate((r_water, obstacle_coms), axis=0)
        _obst_diam = np.concatenate((np.full(Nwater_inserted, fill_value=1.0), np.full(len(_obst_coms), fill_value=obstacle_diameter)))
        r_water_rest  = placing_particles_without_overlapping(N=nwater_rest, Lbox=Lbox, rng=rng, diameter=1.0,
                                                                obstacle_coms=_obst_coms, obstacle_diameter=_obst_diam)
        r_water = np.concatenate((r_water, r_water_rest), axis=0)
        print(f"{nwater_rest} beads are inserted.")

    elif nwater_rest < 0:
        r_water = rng.choice(r_water, size=Ntarget, replace=False)
        print(f"{abs(nwater_rest)} beads are removed.")

    return r_water

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


def placing_polymers(n_seg:int, l_seg:float, d_seg:float, n_poly:int, Lbox:np.ndarray, r_obst, d_obst, seed:int):
    """
    placing spring-bead polymers in a simulation domain.

    Parameter
    ----------
    n_seg (int)   : number of segments consisting of a polymer
    l_seg (float) : length of neighboring segments 
    d_seg (float) : diameter of a segment
    n_poly (int)  : number of polymers to place in a simulation domain
    Lbox (ndarray): edge lenths of a simulation domain
    """
    rng = np.random.default_rng(seed=seed)
    r_polys = []
    bonds = []

    def __random_spherical_point(radius:float):
        _u = rng.random(2)
        _z = -2.0*_u[0] + 1.0
        _x = np.sqrt(1.0-_z*_z)*np.cos(2.0*np.pi*_u[1])
        _y = np.sqrt(1.0-_z*_z)*np.sin(2.0*np.pi*_u[1])
        return np.array([_x, _y, _z])*radius

    obst_exists = r_obst is not None and d_obst is not None

    # place seed particles (1sr segments of polymers)
    r_seeds = placing_particles_without_overlapping(n_poly, Lbox, rng, d_seg)

    for i in range(n_poly):
        r0 = r_seeds[i]
        r_polys.append(r0)
        n_count = 0
        while n_count < n_seg-1:
            # place remaining segments
            ri = r0 + __random_spherical_point(l_seg)

            # test
            # - for other seeds
            accept_seed  = all(__calc_distance_with_pbc(ri, r_seeds, Lbox) > d_seg)
            if accept_seed:
                # - for other polymers
                accept_polys = all(__calc_distance_with_pbc(ri, r_polys, Lbox) > d_seg)
                if accept_polys:
                    r_polys.append(ri)
                    n_count += 1
                    r0 = ri
    
    return r_polys

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
    m = int(np.ceil((L**3 * rho / 4.0)**(1.0 / 3.0)))
    a = L/m

    pos, Lbox = make_lattice(fcc_unit_cell(), (m,m,m), a)

    # sampling
    n = int(rho*L**3)
    rng = np.random.Generator(bit_generator=np.random.MT19937(seed=seed))
    pos_defected = rng.choice(pos, size=n, replace=False)

    return pos_defected, Lbox

