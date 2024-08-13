from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np
import os
flibname = "libflib.so"
base_dir = os.path.dirname(os.path.abspath(__file__))
flibpath = os.path.join(base_dir, "flib")

fortlib = np.ctypeslib.load_library(flibname, flibpath)
fortlib.fort_compute_radial_distribution.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                  ctypes.POINTER(ctypes.c_int),
                                                  ndpointer(dtype=np.float64),
                                                  ctypes.POINTER(ctypes.c_double),
                                                  ndpointer(dtype=np.float64),
                                                  ndpointer(dtype=np.float64)]
fortlib.fort_compute_radial_distribution.restype = None


fortlib.fort_compute_radial_distribution_NP.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                     ctypes.POINTER(ctypes.c_int),
                                                     ctypes.POINTER(ctypes.c_int),
                                                     ndpointer(dtype=np.float64),
                                                     ndpointer(dtype=np.float64),
                                                     ctypes.POINTER(ctypes.c_double),
                                                     ndpointer(dtype=np.float64),
                                                     ndpointer(dtype=np.float64)]
fortlib.fort_compute_radial_distribution_NP.restype = None


def compute_histogram(r:np.ndarray, dr:float, nbins:int, L:np.ndarray):
    natm = len(r)
    hist = np.zeros(nbins)
    fortlib.fort_compute_radial_distribution(ctypes.byref(ctypes.c_int(nbins)), 
                                                 ctypes.byref(ctypes.c_int(natm)), 
                                                 r, 
                                                 ctypes.byref(ctypes.c_double(dr)), 
                                                 L,
                                                 hist)

    return hist

def compute_histogram_NP(core:np.ndarray, r:np.ndarray, dr:float, nbins:int, L:np.ndarray):
    nNP = len(core)
    nSol = len(r) 
    hist = np.zeros(nbins)
    fortlib.fort_compute_radial_distribution_NP(ctypes.byref(ctypes.c_int(nbins)), 
                                                    ctypes.byref(ctypes.c_int(nSol)), 
                                                    ctypes.byref(ctypes.c_int(nNP)), 
                                                    core,
                                                    r,
                                                    ctypes.byref(ctypes.c_double(dr)), 
                                                    L,
                                                    hist)
    return hist


class ComputeRDF:
    """computes Radial Distribution Function (RDF)"""
    def __init__(self, radius:float, nbins:int, ndens:float) -> None:
        """
        Computes Radial Distribution Function (RDF) of a fluid.
        
        Parameters
        -----------
        radius (float): maximum interparticle dinstance. Note that it should not be over a half of box dimension.
        nbins (int): number of bins
        ndens (float): number density in a system. 
        """
        self._radius = radius
        self._nbins = nbins
        self._dr = self._radius/self._nbins
        self._ndens = ndens
        self._bins = np.arange(self._nbins+1)*self._dr
        self._bin_centers = (self._bins[1:]+self._bins[:-1])/2
        self._rdf = np.zeros(nbins)
        self._nstep = 0
        self._natom = 0

    def reset_result(self):
        self._rdf = np.zeros(self._nbins)
        self._nstep = 0

    def compute(self, box:np.ndarray, pos:np.ndarray, pos_NP=None, reset=False):
        """
        compute RDF histogram and add it to the current RDF histogram
        
        Parameters
        ----------
        box (ndarray): system box
        pos (ndaarray): positions of solvents
        pos_NP (ndarray) (optional): positions of center of mass of NPs.
        reset (bool): if true, accumulate rdf

        """

        if self._radius > box.min()/2.0:
            raise ValueError(f"Box is small to compute RDF with given radius. \nradius must be < L/2.")
        
        if isinstance(pos_NP, np.ndarray):
            if pos_NP.shape[1] != 3:
                raise ValueError("pos_NP is an array of the dimension (N,3)")
            self._natom = pos_NP.shape[0]
            _rdf = compute_histogram_NP(pos_NP, pos, self._dr, self._nbins, box)
        else:
            # system for only solvent (monomer) particle
            self._natom = len(pos)
            _rdf = compute_histogram(pos, self._dr, self._nbins, box)
            
        if not reset: 
            self._rdf += _rdf
            self._nstep += 1
        else:
            self._rdf = _rdf

    @property
    def bin_centers(self):
        """ bin centers """
        return self._bin_centers


    @property
    def rdf(self):
        """Radial Distribution Function"""
        _rdf = self._rdf.copy()
        for i, nr in enumerate(_rdf):
            rl = i*self._dr
            rh = rl + self._dr
            const = 4.0/3.0*np.pi*(rh*rh*rh - rl*rl*rl) * self._ndens
            nr /= const
            _rdf[i] = nr
        
        _rdf /= (self._nstep*self._natom)
        return _rdf
    
    @property
    def histogram(self):
        """ histogram"""
        return self._rdf
    