import numpy as np
import math as mt
from typing import Union

class BaseDiscreteParticleObject:
    """
    Base class of the (discrete) particle object.    
    """

    _default_vals = dict()
    _default_vals["verts"] = np.array([0,0,0], dtype=np.float64)
    _default_vals["nvert"] = 0
    _default_vals["vert_types"] = np.array([0], dtype=np.int32)
    _default_vals["vert_type_kinds"] = ["A"]

    _default_vals["bonds"] = np.array([0], dtype=np.int32)
    _default_vals["nbond"] = 0
    _default_vals["bond_r0"] = np.array([0], dtype=np.float64)
    _default_vals["bond_types"] = np.array([0], dtype=np.int32)
    _default_vals["bond_type_kinds"] = ["A"]

    def __init__(self, Nvert: int, 
                       verts:np.ndarray, 
                       vert_typeid:np.ndarray,
                       vert_types:list,
                       Nbond: Union[int, None] = None,
                       bonds: Union[np.ndarray,None] = None,
                       bond_r0:Union[np.ndarray,None] = None,
                       bond_typeid:Union[np.ndarray,None] = None,
                       bond_types:Union[list,None] = None,
                       ) -> None:
        """
        construct particle object. 

        Parameter
        ---------------
        verts (np.ndarray) : coordinates of constituent beads of an object. (Nvert,3)
        vert_types (list) : a list of type-id 
        vert_typeid (np.ndarray) : type id of constituent beads. (Nvert,)
        bonds (np.ndarray) : a list of pairs. (Nbond,2)
        bond_r0 (np.ndarray) : equilibrium distances of pairs. (Nbond,)
        bond_types (list) : a list of type-id for bonds.
        bond_typeid (np.ndarray) : a list of bond-ids. (Nbond,)
        """

        self._nvert = Nvert
        self._verts = self._validate_array(verts, np.float64, [self.nvert,3])
        self._validate_type_kinds(vert_typeid, vert_types)
        self._vert_typeid = self._validate_array(vert_typeid, np.int32, [self.nvert])
        self._vert_types = vert_types

        self._nbond = Nbond
        self._bonds = self._validate_array(bonds, np.int32, [self.nbond,2])
        self._bond_r0 = self._validate_array(bond_r0, np.float64, shape=[self.nbond])
        self._bond_typeid = self._validate_array(bond_typeid, np.int32, shape=[self.nbond])
        self._bond_types = bond_types

    @property
    def nvert(self):
        """number of vertices."""
        return self._nvert

    @property
    def verts(self):
        """
        vertices of particle. datatype is np.float64 and shape is (nvert,3)
        """
        return self._verts

    @property
    def vert_typeid(self):
        """type ids of each vertices. the value must be digit."""
        return self._vert_typeid
    

    @property
    def vert_types(self):
        """
        vertex type kinds. each value of the list is string (such as ["A","B","C",...])
        Each value of the list is corresponding with each digit of `vert_types`. 
        (e.g. "A"=0, "B"=1, ...)
        """
        return self._vert_types
     
    def append_new_type(self, new_type):
        """append new `vert_type_kind` to `vert_type_kinds`."""
        if new_type in self.vert_types:
            raise ValueError(f"Given id {new_type} has already been registered.")

        self._vert_types.append(new_type)

    @property
    def nbond(self):
        """number of bonds"""
        return self._nbond
        
    @property
    def bonds(self):
        """pairs of i- and j-th particles"""
        return self._bonds
    
    @property
    def bond_types(self):
        return self._bond_types

    @property
    def bond_typeid(self):
        return self._bond_typeid

    # validation: 

    def _validate_type_kinds(self, typeids, type_kind_list):
        """
        validate two arrays `vert_types` and `vert_type_kinds` are consistent.
       
        the num of value of `vert_types` (0,1,2,3,...) must match the num of value of `vert_type_kinds`
        """
        if typeids is None or type_kind_list is None:
            return
        else:
            # validate: number of vertex type
            if len(np.unique(typeids)) != len(type_kind_list):
                raise ValueError("Type kinds not matched")

    def _validate_array(self, array:np.ndarray, dt, shape):
        """ validate array. """
        if array is not None:
            array = np.ascontiguousarray(array, dtype=dt)
            array = array.reshape(shape)
        return array


def search_opposed_pair(points:np.ndarray):
    """ search diametrically opposite pairs of particles and compute its distance. """
    rc = points.mean(axis=0)
    # ---- take angle between (v - _c) and z-axis. 
    _norm = np.linalg.norm((points - rc), axis=1)
    _npts = len(points)
    _diam_pairs = []
    for i in range(_npts-1):
        vci = points[i] - rc
        for j in range(i+1, _npts):
            vcj = points[j] - rc
            if mt.isclose(np.dot(vci,vcj)/(_norm[i]*_norm[j]), -1.0):
                _diam_pairs.append([i,j])

    # ---- calculate distance between their particles
    _diam_dists = np.empty(shape=len(_diam_pairs))
    for n, ids in enumerate(_diam_pairs):
        _diam_dists[n] = np.linalg.norm(points[ids[0]] - points[ids[1]])
    
    return _diam_pairs, _diam_dists