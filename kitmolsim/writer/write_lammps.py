import numpy as np


def write_lmp_data(filename:str, Lbox:np.ndarray, 
                   Ntotal:int, Natyp:int, pos:np.ndarray, mole_id:np.ndarray, atype:np.ndarray, charges:np.ndarray, 
                   Nbond:int=None, Nbtyp:int=None, bond_type:np.ndarray=None, bondpair:np.ndarray=None, 
                   bond_coeffs:np.ndarray=None, bond_r0:np.ndarray=None,
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

        has_bond_coeffs = bond_coeffs is not None and bond_r0 is not None

        if has_bond_coeffs:
            f.write("Bond Coeffs\n\n")
            np.savetxt(f, np.vstack((np.arange(Nbtyp)+1, bond_coeffs, bond_r0)).T, fmt="%d %f %f")
    