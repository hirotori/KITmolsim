import unittest
# import context
from kitmolsim.analyze import cluster
import numpy as np

class TestCalcDistance(unittest.TestCase):
    def setUp(self):
        self.L = 5.0
        self.Lbox = np.array([self.L]*3)
        # atom position (-L/2 <= x,y,z <= L/2)
        self.pos = np.array(
            [[0.0, 0.0, 0.0],
             [2.5, 0.0, 0.0],
             [-2.5, 0.0, 0.0],
             [2.5, 0.0, 2.5]]
        )

    def test_calc_distance(self):
        mat = cluster.calc_distance_with_pbc(self.pos, self.Lbox)

        print(mat)
        self.assertAlmostEqual(mat[0,0], 0.0, delta=1e-9)
        self.assertAlmostEqual(mat[0,1], 2.5, delta=1e-9)
        self.assertAlmostEqual(mat[0,2], 2.5, delta=1e-9)
        self.assertAlmostEqual(mat[0,3], 2.5*np.sqrt(2), delta=1e-9)

        self.assertAlmostEqual(mat[1,0], 2.5, delta=1e-9)
        self.assertAlmostEqual(mat[1,1], 0.0, delta=1e-9)
        self.assertAlmostEqual(mat[1,2], 0.0, delta=1e-9)
        self.assertAlmostEqual(mat[1,3], 2.5, delta=1e-9)
        
        self.assertAlmostEqual(mat[2,0], 2.5, delta=1e-9)
        self.assertAlmostEqual(mat[2,1], 0.0, delta=1e-9)
        self.assertAlmostEqual(mat[2,2], 0.0, delta=1e-9)
        self.assertAlmostEqual(mat[2,3], 2.5, delta=1e-9)
        
        self.assertAlmostEqual(mat[3,0], 2.5*np.sqrt(2), delta=1e-9)
        self.assertAlmostEqual(mat[3,1], 2.5, delta=1e-9)
        self.assertAlmostEqual(mat[3,2], 2.5, delta=1e-9)
        self.assertAlmostEqual(mat[3,3], 0.0, delta=1e-9)
        
    
if __name__ == "__main__":
    unittest.main()
