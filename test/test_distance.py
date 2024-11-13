import unittest
from kitmolsim.analyze import _dist
import numpy as np

class test_PlacingPolymers(unittest.TestCase):
    def test_calc_distance(self):
        x0 = np.array([0.0, 0.0, 0.0])
        y0 = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
        Lbox = np.array([10.0, 10.0, 10.0])

        dist = _dist.calc_distance_with_p2_pbc(x0, y0, Lbox)
        for d in dist:
            self.assertAlmostEqual(d, 1.0, delta=1e-9)

    def test_calc_distance_overlap(self):
        x0 = np.array([-5.0, 0.0, 0.0])
        y0 = np.array([[1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [1.0, 0.0, 1.0]])
        Lbox = np.array([10.0, 10.0, 10.0])

        dist = _dist.calc_distance_with_p2_pbc(x0, y0, Lbox)
        exact = np.array([4.0, np.sqrt(17.0), np.sqrt(17.0)])
        for d, d_true in zip(dist, exact):
            self.assertAlmostEqual(d, d_true, delta=1e-9)


if __name__ == "__main__":
    unittest.main()