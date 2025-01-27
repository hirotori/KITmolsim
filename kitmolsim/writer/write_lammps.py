import numpy as np
from ..reader import read_lammps as rlmp

def write_lmp_data(filename:str, Lbox:np.ndarray, 
                   Ntotal:int, Natyp:int, pos:np.ndarray, mole_id:np.ndarray, atype:np.ndarray, charges:np.ndarray, 
                   Nbond:int=None, Nbtyp:int=None, bond_type:np.ndarray=None, bondpair:np.ndarray=None, 
                   bond_params:np.ndarray=None,
                   mode="w"):
    """
    write data as lammps data format.   

    atom_style is "full". molecular-id and charges are needed.

    """
    with open(file=filename, mode=mode) as f:
        f.write("from initCond.py\n\n")
        f.write(f"{Ntotal} atoms\n")
        f.write(f"{Natyp} atom types\n\n")
        if Nbond is not None: f.write(f"{Nbond} bonds\n")
        if Nbtyp is not None: f.write(f"{Nbtyp} bond types\n\n")
        f.write(f"{-Lbox[0]/2} {Lbox[0]/2} xlo xhi\n")
        f.write(f"{-Lbox[1]/2} {Lbox[1]/2} ylo yhi\n")
        f.write(f"{-Lbox[2]/2} {Lbox[2]/2} zlo zhi\n")
        f.write("\n")
        f.write("Atoms\n\n")
        atomid = np.arange(Ntotal) + 1
        np.savetxt(f, np.column_stack((atomid, mole_id, atype, charges, pos)), fmt=['%7.0f', '%7.0f', '%2.0f', '%2.0f', '%13.8f', '%13.8f', '%13.8f'])
        #np.savetxt("bond.dat", np.column_stack((bondid, btypeid, bonds_list)), fmt=['%7.0f', '%7.0f', '%7.0f', '%7.0f'])
        f.write("\n")

        has_bond_group = Nbond is not None and Nbtyp is not None and bond_type is not None and bondpair is not None

        if has_bond_group:
            f.write("Bonds\n\n")
            bondids = np.arange(Nbond)
            np.savetxt(f, np.vstack((bondids, bond_type, bondpair[:,0], bondpair[:,1])).T, fmt="%d %d %d %d")
            f.write("\n")

        has_bond_coeffs = bond_params is not None

        if has_bond_coeffs:
            f.write("Bond Coeffs\n\n")
            if bond_params.shape[1] != Nbtyp:
                raise ValueError("the 2nd dimension of bond_params must be equal to Nbtyp.")
            fmt = ["%d"]
            fmt.extend(["%f"]*bond_params.shape[0])
            np.savetxt(f, np.vstack((np.arange(Nbtyp)+1, bond_params)).T, fmt=fmt)
    

class LAMMPSTrajectoryWriter:
    
    frames_: list[rlmp.LAMMPSTrajectoryFrame] = []

    def __init__(self, filename:str) -> None:
        self.frames_ = []
        self.filename = filename
        
    def append(self, frame:rlmp.LAMMPSTrajectoryFrame):
        self.frames_.append(frame)

    def save(self):
        with open(self.filename, mode="w") as f:
            for frame in self.frames_:
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{frame.timestep}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{frame.natom}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                for n in range(3):
                    f.write(f"{-frame.box[n]/2} {frame.box[n]/2}\n")
                f.write("ITEM: ATOMS id type xu yu zu\n")
                ids = np.arange(frame.natom)+1
                np.savetxt(f, X=np.column_stack((ids, frame.atype, frame.pos)), 
                           fmt="%d %d %f %f %f")
        f.close()
