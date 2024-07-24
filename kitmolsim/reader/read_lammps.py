import numpy as np
import collections.abc
import typing
import warnings

class LAMMPSTrajectoryFrame:
    """
    A snapshot of trajectory.
    This contains:
    - timestep: timestep
    - natom: number of atoms
    - atype: atom types ((natom,) ndarray)
    - ids: atom ids ((natom,) ndarray)
    - pos: atom positions ((natom,3) ndarray)
    - box: system box dimensions ((6,) ndarray)
    """
    def __init__(self, timestep:int, natom:int, 
                 ids:np.ndarray, 
                 atype:np.ndarray, 
                 pos:np.ndarray, 
                 box:np.ndarray,
                 vel:np.ndarray = None,
                 img:np.ndarray = None):
        self.timestep: int = timestep
        self.natom: int = natom
        self.atype: np.ndarray = atype
        self.ids: np.ndarray = ids
        self.pos: np.ndarray = pos
        self.box: np.ndarray = box
        self.vel = None
        self.img = None

        self.validate()

    def validate(self):
        if self.atype is not None: _ = self.atype.reshape([self.natom])
        if self.ids is not None: _ = self.ids.reshape([self.natom])
        if self.pos is not None: _ = self.pos.reshape([self.natom,3])
        if self.vel is not None: _ = self.vel.reshape([self.natom,3])
        if self.img is not None: _ = self.img.reshape([self.natom,3])

class LAMMPSDataFrame(LAMMPSTrajectoryFrame):
    def __init__(self, timestep: int, natom: int, 
                 ids: np.ndarray, atype: np.ndarray, 
                 pos: np.ndarray, box: np.ndarray, 
                 vel: np.ndarray = None, img: np.ndarray = None, 
                 nbond: int = None, nbond_type: int = None, 
                 bonds: np.ndarray = None, bondtypes: np.ndarray=None, 
                 bond_coeffs : np.ndarray = None, bond_r0 : np.ndarray = None):
        
        super().__init__(timestep, natom, ids, atype, pos, box, vel, img)
        self.nbond = nbond
        self.nbond_type = nbond_type
        self.bonds = bonds
        self.bondtypes = bondtypes
        self.bond_coeffs = bond_coeffs
        self.bond_r0 = bond_r0

        self.validate_additional_properties()

    def validate_additional_properties(self):
        if self.nbond is not None:
            _ = self.bonds.reshape([self.nbond,2])
            _ = self.bondtypes.reshape([self.nbond])
            _ = self.bond_coeffs.reshape([self.nbond_type])
            _ = self.bond_r0.reshape([self.nbond_type])

class LAMMPSTrajectory(collections.abc.Container):
    """
    LAMMPS trajectory containing multiple frames.
    """

    def __init__(self):
        self.frames: typing.List[LAMMPSTrajectoryFrame] = []
        
    def append(self, frame:LAMMPSTrajectoryFrame):
        if not isinstance(frame, LAMMPSTrajectoryFrame):
            raise TypeError("frame must be LAMMPSTrajectoryFrame")
        self.frames.append(frame)

    def __iter__(self):
        yield from self.frames

    def __len__(self):
        return len(self.frames)
    
    def __contains__(self, __x: object) -> bool:
        return __x in self.frames
    
    def __getitem__(self, key) -> LAMMPSTrajectoryFrame:
        return self.frames[key]
    
    def unwrap(self) -> None:
        """ 
        unwraps all coordinates contained in frames.

        All positions except the initial position are to be unwrapped.
        """
        _nframe = len(self.frames)
        if _nframe <= 1:
            raise ValueError(f"trajectory does not have enough frames for coordinates to be unwrapped.")
        
        _natom = self.frames[0].natom
        for i in range(_nframe-1):
            _unwrap_pos(_natom, 
                        prev_pos=self.frames[i].pos, 
                        next_pos=self.frames[i+1].pos,
                        box=self.frames[i].box)
            
    def wrap(self):
        """ wrap the unwrapped coordinates"""
        for snap in self.frames:
            L = snap.box
            dL = L/2
            snap.pos = (snap.pos + dL)%L - dL


def _unwrap_pos(natom, prev_pos:np.ndarray, next_pos:np.ndarray, box:np.ndarray):
    """ 
    unwrao corrdinates from the previous position.
    """
    dL = box/2
    for i in range(natom):
        dx = next_pos[i,:] - prev_pos[i,:]
        if dx[0] > dL[0]:
            next_pos[i,0] -= ( dx[0]+dL[0])//box[0]*box[0]
        if dx[0] < -dL[0]:
            next_pos[i,0] += (-dx[0]+dL[0])//box[0]*box[0]
    

def read_dumpfile(filename:str):
    """
    read dump file written by the command "dump" with the keyword option "atom". 

    Parameter
    ----------
    filename: str
        filename (.lammpstrj)
    """
    trajectory = LAMMPSTrajectory()
    n = 0
    with open(filename, mode="r") as f:

        keyword = f.readline() #read first line
        while (keyword):
            if "ITEM: TIMESTEP" in keyword:
                timestep = int(f.readline().split(" ")[0])

            if "ITEM: NUMBER OF ATOMS" in keyword:
                natom = int(f.readline().split(" ")[0])
                
            if "ITEM: BOX BOUNDS" in keyword:
                lxo, lxh = [float(item) for item in f.readline().split()]
                lyo, lyh = [float(item) for item in f.readline().split()]
                lzo, lzh = [float(item) for item in f.readline().split()]
                box = np.array([lxh-lxo, lyh-lyo, lzh-lzo])

            if "ITEM: ATOMS" in keyword:
                # count number of items (e.g. "ITEM: ATOMS type id x y z ...")
                nitem = len(keyword.split(" ")[2:])
                # For simplicity, assume the format is "atom" and "custom" with 5 tags (type, id, coordinates (x,y,z or xs,ys,zs or xu,yu,zu))
                assert(nitem == 5)
                buffer = np.loadtxt(f, max_rows=natom)
                
                _frame = LAMMPSTrajectoryFrame(timestep, natom, buffer[:,0], buffer[:,1], buffer[:,2:5], box)

                trajectory.append(_frame)
                print(f"\r Number of item = {n}", end="")
                n += 1

            keyword = f.readline() # read the first line of next frame
    print("")
    return trajectory
    
def read_datafile_chunk(filename:str, ntotal:int):
    with open(file=filename, mode="r") as f:
        f.readline() #the first line (comment)
        buffer = [words for words in f.readline().split(" ")] # the second line ()
        ncol1 = len(buffer[1:])
        buffer = [words for words in f.readline().split(" ")] # the third line ()
        ncol2 = len(buffer[1:])
        assert(ncol1 == 3)
        assert(ncol2 == 4)

        # *** load main data ***
        i = 1
        val_all = []
        while (i <= ntotal):
            try:
                _, nchunk, _ = np.loadtxt(f, max_rows=1, unpack=True)
                id, x, natom, val = np.loadtxt(f, max_rows=int(nchunk), unpack=True)
                val_all.append(val)
                
                i += 1
                print(f"\r data count: {i}/{ntotal}", end="")
            except:
                warnings.warn(f"load_datafile_chunk:: Number of the data in this file is less than {ntotal}.")
                break
        print("")
        return np.array(x), np.array(val_all)


def read_datafile(filename:str):
    """
    read data file that read by the lammps command 'read_data'. 
    
    file format: https://docs.lammps.org/99/data_format.html
    """
    
    with open(file=filename, mode="r") as f:
        # 
        comment = f.readline() # the 1st line (comment)
        keyword = f.readline() # new_line
        atoms = f.readline().split(" ")
        natom = int(atoms[0])
        n = 0
        keyword = f.readline()

        nbond = None
        nangle = None
        ndihed = None
        nimpro = None
        
        nbond_type = None
        nangle_type = None
        ndihed_type = None

        bondid = None
        bondtype_id = None
        bond_pair = None

        bondtype_id = None
        bond_coeffs = None
        bond_r0     = None
        while (keyword):
            
            words = keyword.split(" ")

            # box
            if len(words) == 4:
                if words[2] == "xlo" and words[3] == "xhi\n":
                    xlo = float(words[0])
                    xhi = float(words[1])
                if words[2] == "ylo" and words[3] == "yhi\n":
                    ylo = float(words[0])
                    yhi = float(words[1])
                if words[2] == "zlo" and words[3] == "zhi\n":
                    zlo = float(words[0])
                    zhi = float(words[1])
            
            # bond, angles, etc.
            if len(words) == 2:
                if words[1] == "bonds\n":
                    nbond = int(words[0])
                if words[1] == "angles\n":
                    nangle = int(words[0])
                if words[1] == "dihedrals\n":
                    ndihed = int(words[0])
                if words[1] == "impropers\n":
                    nimpro = int(words[0])
            
            # types
            if len(words) == 3:
                if words[1] == "atom":
                    natom_type = int(words[0])
                if words[1] == "bond":
                    nbond_type = int(words[0])
                if words[1] == "angle":
                    nangle_type = int(words[0])
                if words[1] == "dihedral":
                    ndihed_type = int(words[0])

            # Atoms etc.
            if keyword == "Atoms\n":
                _ = f.readline()
                buffer = np.loadtxt(f, max_rows=natom)
                atomid = buffer[:,0]
                molecule_id = buffer[:,1]
                atom_type   = buffer[:,2]
                q           = buffer[:,3]
                pos         = buffer[:,4:7]

            if keyword == "Bonds\n":
                _ = f.readline()
                buffer = np.loadtxt(f, max_rows=nbond)
                bondid = buffer[:,0]
                bondtype_id = buffer[:,1]
                bond_pair = buffer[:,2:4]

            if keyword == "Bond Coeffs\n":
                _ = f.readline()
                buffer = np.loadtxt(f, max_rows=nbond_type)
                bondtype_id_= buffer[:,0] if nbond_type > 1 else buffer[0]
                bond_coeffs = buffer[:,1] if nbond_type > 1 else buffer[1]
                bond_r0     = buffer[:,2] if nbond_type > 1 else buffer[2]

                
            keyword = f.readline() #read next line (exit if None)

        box = np.array([xhi-xlo, yhi-ylo, zhi-zlo], dtype=np.float32)
        return LAMMPSDataFrame(timestep=0, natom=natom, ids=atomid, 
                                atype=atom_type, pos=pos, box=box, 
                                nbond=nbond, nbond_type=nbond_type, 
                                bonds=bond_pair, bondtypes=bondtype_id, 
                                bond_coeffs=bond_coeffs, bond_r0=bond_r0)
    

