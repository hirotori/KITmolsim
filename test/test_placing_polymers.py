import unittest
from kitmolsim.init import placing_polymers
from kitmolsim.writer import write_cdview
from scipy.spatial.distance import pdist
import numpy as np

class test_PlacingPolymers(unittest.TestCase):
    def setUp(self):
        self.L = 30.0
        self.Lbox = np.full(3, fill_value=self.L)
        self.n_seg = 20
        self.l_seg = 1.0
        self.d_seg = 1.0
        self.n_poly = 5

    def test_placing_polymers(self):
        r_poly = placing_polymers(self.n_seg, self.l_seg, d_seg=self.d_seg, n_poly=self.n_poly, Lbox=self.Lbox, 
                                  r_obst=None, d_obst=None, seed=100)
        bonds = np.concatenate(tuple(np.array([[i,i+1] for i in range(self.n_seg-1)])+j*self.n_seg for j in range(self.n_poly)), axis=0)
        atypeid = np.concatenate(tuple(np.full(self.n_seg, fill_value=i) for i in range(self.n_poly)))

        # shape
        self.assertEqual(r_poly.shape[0], self.n_poly*self.n_seg)
        self.assertEqual(r_poly.shape[1], 3)

        # distance
        dist = pdist(r_poly)
        self.assertTrue(all(dist > self.d_seg))

        box_s = tuple(-l*0.5 for l in self.Lbox)
        box_e = tuple( l*0.5 for l in self.Lbox)
        write_cdview.write_cdv("test.cdv", atom_pos=np.array(r_poly), atom_type=atypeid,
                            box_s=box_s, box_e=box_e, bondpair=bonds, bondtypeid=np.zeros(bonds.shape[0]))


if __name__ == "__main__":
    unittest.main()