# import context
import unittest
from kitmolsim.shape import util
import numpy as np

class test_BaseParticleObject(unittest.TestCase):
    def test_initalize(self):
        # create object
        x = np.array([[0,0,0]], dtype="float64")
        types = np.arange(1)
        type_kinds = ["A"]
        obj = util.BaseParticleObject(x, types, type_kinds)

        # types mismatch
        types = np.arange(2)
        with self.assertRaises(ValueError):
            obj = util.BaseParticleObject(x, types, type_kinds)

        # type kinds mismatch
        types = np.arange(1)
        type_kinds += ["B"]
        with self.assertRaises(ValueError):
            obj = util.BaseParticleObject(x, types, type_kinds)



if __name__ == "__main__":
    unittest.main()
