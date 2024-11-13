import unittest
from kitmolsim.init import placing_polymers
from kitmolsim.analyze import _dist
from scipy.spatial.distance import pdist
import numpy as np

class test_PlacingPolymers(unittest.TestCase):
    def setUp(self):
        self.L = 30.0
        self.Lbox = np.full(3, fill_value=self.L)
        self.n_seg = 10
        self.l_seg = 1.0
        self.d_seg = 1.0
        self.n_poly = 100

    def test_placing_polymers(self):
        r_poly = placing_polymers(self.n_seg, self.l_seg, d_seg=self.d_seg, n_poly=self.n_poly, Lbox=self.Lbox, 
                                  r_obst=None, d_obst=None, seed=100)

        # shape
        self.assertEqual(r_poly.shape[0], self.n_poly*self.n_seg)
        self.assertEqual(r_poly.shape[1], 3)

        # bonds = np.concatenate(tuple(np.array([[i,i+1] for i in range(self.n_seg-1)])+j*self.n_seg for j in range(self.n_poly)), axis=0)
        # atypeid = np.concatenate(tuple(np.full(self.n_seg, fill_value=i) for i in range(self.n_poly)))
        # box_s = tuple(-l*0.5 for l in self.Lbox)
        # box_e = tuple( l*0.5 for l in self.Lbox)
        # write_cdview.write_cdv("test.cdv", atom_pos=np.array(r_poly), atom_type=atypeid,
        #                     box_s=box_s, box_e=box_e, bondpair=bonds, bondtypeid=np.zeros(bonds.shape[0]))

        # distance
        dist = pdist(r_poly)
        self.assertTrue(all(dist > 0.0))
        self.assertTrue(all(dist >= self.d_seg))

    def test_placing_polymers_obst(self):
        r_obst = np.array([[0.0, 0.0, 0.0]])
        R = 3.0
        d_obst = np.array([[R*2.0]])
        r_poly = placing_polymers(self.n_seg, self.l_seg, d_seg=self.d_seg, n_poly=self.n_poly, Lbox=self.Lbox, 
                                  r_obst=r_obst, d_obst=d_obst, seed=100)

        self.assertEqual(r_poly.shape[0], self.n_poly*self.n_seg)
        self.assertEqual(r_poly.shape[1], 3)

        dist = pdist(r_poly)
        self.assertTrue(all(dist > 0.0))
        self.assertTrue(all(dist >= self.d_seg))

        dist = _dist.calc_distance_with_p2_pbc(r_obst[0], r_poly, self.Lbox)
        self.assertTrue(all(dist > 0.0))
        self.assertTrue(all(dist >= self.d_seg/2.0+R))


if __name__ == "__main__":
    unittest.main()