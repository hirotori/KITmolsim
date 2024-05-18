from kitmolsim import init
from kitmolsim.writer import write_cdview
import numpy as np

pos, box = init.make_lattice(unit_lattice=init.fcc_unit_cell(), num_cell=(10,10,10), lattice_constant=0.75)

atype = np.zeros(len(pos))
write_cdview.write_cdv("test_fcc.cdv", atom_pos=pos, atom_type=atype, box_s=-box/2, box_e=box/2)


pos, box = init.make_defected_fcc(L=5.0, rho=0.75, seed=42)
atype = np.zeros(len(pos))
write_cdview.write_cdv("test_dfcc.cdv", atom_pos=pos, atom_type=atype, box_s=-box/2, box_e=box/2)
