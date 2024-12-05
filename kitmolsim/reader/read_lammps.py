import numpy as np
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
        if self.box is not None: self.box = self.box.reshape([3]) 
        if self.atype is not None: self.atype = self.atype.reshape([self.natom])
        if self.ids is not None: self.ids = self.ids.reshape([self.natom])
        if self.pos is not None: self.pos = self.pos.reshape([self.natom,3])
        if self.vel is not None: self.vel = self.vel.reshape([self.natom,3])
        if self.img is not None: self.img = self.img.reshape([self.natom,3])

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

class LAMMPSTrajectory:
    """
    Class to handle LAMMPS dump files efficiently.
    Instead of loading the entire file, only the required timesteps are parsed.

    Note
    ---------
    The region definition is valid only for the periodic boundary condition (`box bounds pp pp pp`). 
    The atoms are assumed to be arranged in the following format: 
    atom ids, atom types, and coordinates (x,y,z or xu,yu,zu).


    Example
    ---------
    >>> dat_file = LAMMPSTrajectory("test.dat")
    >>> frame = dat_file[10]  # Retrieve data for the 10th frame
    >>> print("Positions:", frame.pos)
    >>> print("Box size:", frame.box)
    >>> print("Particle types:", frame.atype)

    """
    def __init__(self, filename:str):
        """
        Class to handle LAMMPS dump files efficiently.
        Instead of loading the entire file, only the required timesteps are parsed.
        
        Parameters
        -----------
        filename (str): Path to the LAMMPS dump file.   
        """

#        self.frames: typing.List[LAMMPSTrajectoryFrame] = []
        self.filename = filename
        self._index_map = self._build_index()
        self.nframe = len(self._index_map) - 1


    def _build_index(self):
        """
        Build an index for the file.
        Maps each timestep to its starting position in the file.
        
        Returns:
            list: A list where values are file positions.

        Note:
            If file size is larger than the system memory size, it does not work succesfully.
        """
        import re
        index_map = []
        pattern = re.compile(r"ITEM: TIMESTEP\n(\d+)\n")
        with open(self.filename, 'r') as f:
            content = f.read()  # Read the entire file into memory
            for match in pattern.finditer(content):
                index_map.append(match.start())
        return index_map

    def _read_frame(self, position):
        """
        Read the data of a single timestep from the specified file position.
        
        Parameters
        ------------
            position (int): Position in the file where the timestep data starts.
        
        Returns:
            DataFrame: A DataFrame object containing the data of the specified timestep.
        """
        with open(self.filename, 'r') as f:
            f.seek(position)
            assert f.readline().startswith("ITEM: TIMESTEP")
            timestep = int(f.readline().strip())  

            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            num_atoms = int(f.readline().strip())

            assert f.readline().startswith("ITEM: BOX BOUNDS pp pp pp")
            box_bounds = []
            for _ in range(3):
                box_bounds.append([float(x) for x in f.readline().strip().split()])
            box_bounds = np.array([l[1] - l[0] for l in box_bounds])

            assert f.readline().startswith("ITEM: ATOMS")
            # For simplicity, assume the format is "atom" and "custom" with 5 tags 
            # (type, id, coordinates (x,y,z or xs,ys,zs or xu,yu,zu))
            buffer = np.loadtxt(f, max_rows=num_atoms)
            atom_ids   = buffer[:,0]
            atom_types = buffer[:,1]
            positions = buffer[:,2:5]

        return LAMMPSTrajectoryFrame(timestep=timestep, 
                                     natom=num_atoms, 
                                     ids=atom_ids, 
                                     pos=positions, 
                                     box=box_bounds, 
                                     atype=atom_types)


    def __len__(self):
        return self.nframe
    
    
    def __getitem__(self, timestep):
        """
        Retrieve the data of the specified timestep.
        
        Args:
            timestep (int): The timestep number to retrieve.
        
        Returns:
            DataFrame: A DataFrame object containing the data of the specified timestep.
        
        Raises:
            KeyError: If the specified timestep is not found in the file.

        Note: 
            Slice index is not supported.
        """
        if timestep not in self._index_map:
            raise KeyError(f"Timestep {timestep} not found in the file.")
        position = self._index_map[timestep]
        return self._read_frame(position)
    

def read_dumpfile(filename:str):
    """
    This is deprecated. 

    read dump file written by the command "dump" with the keyword option "atom". 

    Parameter
    ----------
    filename: str
        filename (.lammpstrj)
    """
    warn_msg = "This procedure is deprecated. Call a `LAMMPSTrajectory` instance instead."
    warnings.warn(warn_msg, DeprecationWarning)

    return LAMMPSTrajectory(filename)

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
    

