import unittest
from kitmolsim.shape import icomesh
from kitmolsim.analyze import inertia
import numpy as np

class TestInertia(unittest.TestCase):
    def test_inertia_sphere_shell(self):
        shell = icomesh.Icosphere(radius=3)
        I = inertia.compute_inertia_tensor(shell.vertices, np.ones(shell.nvert))

        # Ixx = Iyy = Izz = 2/3*M*r^2, Ixy = Iyz = Izx = 0
        I_true = 2./3.*1.0*shell.nvert*3.0**2.0

        self.assertAlmostEqual(I[0,0], I_true)
        self.assertAlmostEqual(I[1,1], I_true)
        self.assertAlmostEqual(I[2,2], I_true)
        self.assertAlmostEqual(I[0,1], 0.0   )
        self.assertAlmostEqual(I[0,2], 0.0   )
        self.assertAlmostEqual(I[1,0], 0.0   )
        self.assertAlmostEqual(I[1,2], 0.0   )
        self.assertAlmostEqual(I[2,0], 0.0   )
        self.assertAlmostEqual(I[2,1], 0.0   )
        

if __name__ == "__main__":
    unittest.main()
