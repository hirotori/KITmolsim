import numpy as np
from kitmolsim import init
from scipy.spatial import distance, distance_matrix
Lbox = [14.14,14.14,14.14]
D =0.9
pos, vel = init.create_random_state(N=2000, Lbox=Lbox, kT=1.0, seed=42, overlap=False, diameter=D)
d = distance.pdist(pos)

assert(np.count_nonzero(d < D) == 0)