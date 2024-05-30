import context
import unittest
from kitmolsim.shape import sphere
import numpy as np

class TestIcosphere(unittest.TestCase):
    def setUp(self) -> None:
        self.sphere = sphere.IcosphereParticle(radius=3.0)
        ids = np.empty(0, dtype=np.int32)
        for _d in self.sphere.bond_group_ids:
            ids = np.concatenate((ids,_d))
        self.ids = np.sort(ids)

    def test_property(self):
        self.assertEqual(self.sphere.nvert, 162)
        self.assertEqual(self.sphere.nbond, 561)
        self.assertEqual(self.sphere.nbond_group, 6)
        
        self.assertTrue(np.array_equal(self.ids, np.arange(self.sphere.nbond)))    # must be True


    
if __name__ == "__main__":
    unittest.main()
