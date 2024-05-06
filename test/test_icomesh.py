import unittest
import context
from kitmolsim.shape import icomesh

class TestIcosphere(unittest.TestCase):
    def test_init(self):
        self.sphere = icomesh.Icosphere(radius=3.0)

    
if __name__ == "__main__":
    unittest.main()
