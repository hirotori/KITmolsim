import context
import unittest
from kitmolsim.shape import sphere

class TestIcosphere(unittest.TestCase):
    def setUp(self) -> None:
        self.sphere = sphere.IcosphereParticle(radius=3.0)

    def test_property(self):
        self.assertEqual(self.sphere.nvert, 162)
        self.assertEqual(self.sphere.nbond_group, 6)

    
if __name__ == "__main__":
    unittest.main()
