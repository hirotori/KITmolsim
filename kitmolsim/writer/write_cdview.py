import numpy as np
import os
from typing import Union, Tuple


def write_cdv(filename:str,
              atom_pos:np.ndarray, atom_type:np.ndarray, 
              box_s:Tuple[float], 
              box_e:Tuple[float], 
              box_wt=0.01,
              bondpair=None,
              bondtypeid=None,
              radius:Union[None,np.ndarray] = None,
              color :Union[None,np.ndarray] = None,
              light_pos=[1.2,1.0,1.1]):
    """
    write out simulation data as common cdview format (.cdv).

    Parameter
    ---------
    filename (str) : file name
    timestep (int) : time step 
    atom_pos (np.ndarray) : atom coordinates. shape = (N,3)
    atom_type (np.ndarray) : atom types. shape = (N,)
    box_s (tuple(float)) : coordinate (x,y,z) of the starting point of simulation box. 
    box_e (tuple(float)) : coordinate (x,y,z) of the ending point of simulation box.
    box_wt (float,optional) : the line width of the simulation box. box_wt = 0.01 by default.
    bondpair (np.ndarray) : bond information. shape = (nbond,2) where nbond is the total number of bonds.
    bondtypeid (np.ndarray) : bond type-id. if `bondtypeid == None`, then all bond types are interpreted as a same id "0".
    radius (np.ndarray) : a sequence of radii for each atom type
    color (np.ndarray) : a sequence of colors for each atom type (R,G,B)
    light_pos (tuple(float)) : lite source position in a screen
    """

    __validate(atom_pos, atom_type, box_s, box_e, bondpair, bondtypeid, radius, color, light_pos)

    with open(file=os.path.expanduser(filename), mode="w") as f:
        f.write(f"# box_sx={box_s[0]} box_sy={box_s[1]} box_sz={box_s[2]}\n")
        f.write(f"# box_ex={box_e[0]} box_ey={box_e[1]} box_ez={box_e[2]}\n")
        f.write(f"# box_wt={box_wt}\n")


        if isinstance(radius, np.ndarray): 
            buff = [f"r{i}={r}" for i, r in enumerate(radius)]
            f.write("# ")
            print(*buff, sep=" ", file=f)

        if isinstance(color, np.ndarray): 
            buff = [f"c{i}=({c[0]},{c[1]},{c[2]})" for i, c in enumerate(color)]
            f.write("# ")
            print(*buff, sep=" ", file=f)

        # bond information is first inserted (if exist)
        if bondpair is not None:
            header_strings = np.full(shape=len(bondpair), fill_value="CDVIEW_BOND")
            if bondtypeid is not None:
                np.savetxt(f, X=np.column_stack((header_strings, bondpair.astype("i4"), bondtypeid.astype("i4"))), fmt="%s")
            else:
                np.savetxt(f, X=np.column_stack((header_strings,bondpair.astype("i4"))), fmt="%s")
        
        # particle information
        ids = np.arange(atom_pos.shape[0])
        np.savetxt(f, X=np.column_stack((ids, atom_type, atom_pos)), fmt="%d %d %f %f %f")


def __validate(pos, atype, bs, be, bpair, btypeid, rad, c, lpos):
    """
    validate given arguments
    """
    n, nd = pos.shape
    if nd != 3:
        raise ValueError(f"size of the rank 2 must be 3 but {nd} is given")
    _ = atype.reshape([n]).astype(np.int32) # test if shape = (N,)

    if bs is not None: assert(len(bs) == 3)
    if be is not None: assert(len(be) == 3)
    
    if bpair is not None: 
        npair = len(bpair)
        _ = bpair.reshape([npair,2])
        min_atomid = bpair.min()
        max_atomid = bpair.max()
        # if min_atomid != 0:
        #     raise ValueError("atom id in bondpair not started from 0.")

    if btypeid is not None: _ = btypeid.reshape([npair])

    n_atype = len(np.unique(atype))

    if rad is not None: _ = rad.reshape([n_atype])
    if c   is not None: _ = c.reshape([n_atype,3])

    if lpos is not None: assert(len(lpos) == 3)
