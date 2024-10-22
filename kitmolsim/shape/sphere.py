from . import icomesh, util
import numpy as np

class IcosphereParticle(util.BaseDiscreteParticleObject):    
    def __init__(self, radius:float, center=[0,0,0], nsub=2, bond_type="diametric", default_vert_type="A") -> None:
        icos = icomesh.Icosphere(radius, center, nsub)
        self._radius = radius
        self._center = center
        self._bond_type = bond_type
        
        # create neighbor list
        point_connectivity = [np.array(icos.pt_neighbors[vertId]) for vertId in range(icos.nvert)]
        
        # construct a list of pairs of nearest neighbor particles on a surface with no duplicates
        surf_pairs = []
        for ptId, connIds in enumerate(point_connectivity):
            # Among connIds, only the id greater than ptId should be examined.
            for pairId in connIds[connIds > ptId]:
                surf_pairs.append([ptId, pairId])

        # compute distances between nearest neighbors on surface
        surf_pairs = np.array(surf_pairs)
        surf_pair_distances = np.zeros(len(surf_pairs))
        for pairId in range(len(surf_pairs)):
            surf_pair_distances[pairId] = np.linalg.norm(icos.vertices[surf_pairs[pairId,0]] - icos.vertices[surf_pairs[pairId,1]])

        # construct bond array (inner)
        if bond_type == "diametric":
            _diam_pairs, _diam_dists = util.search_opposed_pair(icos.vertices)
            verts = icos.vertices
            bonds = np.concatenate((surf_pairs, _diam_pairs), axis=0)
            bond_distances = np.concatenate((surf_pair_distances, _diam_dists))

        elif bond_type == "center":
            #pairs of center point (id = Nverts) and vertices on surface (id=0~Nverts-1)
            Pt2centers = np.array([[i, self.nvert] for i in range(self.nvert)]) 

            N_center_bonds = self.nvert
            center_distances = np.zeros(N_center_bonds)
            for Id in range(N_center_bonds):
                center_distances[Id] = np.linalg.norm(icos.vertices[Id] - center)
            
            verts = np.vstack((icos.vertices, center))
            bonds = np.concatenate((surf_pairs, Pt2centers), axis=0)
            bond_distances = np.concatenate((surf_pair_distances, center_distances))

        else:
            # no inner bonds constructed
            bonds = surf_pairs
            bond_distances = surf_pair_distances
        
        # properties required for molecular simulation
        # ** vertex
        vert_typeid = np.zeros(verts.shape[0])
        vert_types = [default_vert_type]

        # ** bonds
        bond_r0 = bond_distances.round(decimals=5) 
        _r0s, _ = np.unique(bond_r0, return_counts=True) # unique r0
        bond_group_id_list = [] # list of bonds in different group
        nbond_group = len(_r0s)
        bond_group_r0 = _r0s
        bond_typeid = np.zeros(bonds.shape[0], dtype=np.int32) # 0 for surface pairs
        bond_types = [f"bond{id}" for id in range(nbond_group)]

        for _bt, _r0 in enumerate(_r0s):
            bond_group_ids = np.where(bond_r0 == _r0)[0]
            bond_group_id_list.append(bond_group_ids)
            bond_types[bond_group_ids] = _bt

        super().__init__(len(verts), verts, vert_typeid, vert_types,
                         len(bonds), bonds, bond_r0, bond_typeid, bond_types)

        # bond group
        self.nbond_group = nbond_group
        self.bond_groups = bond_group_id_list
        self.bond_group_r0 = bond_group_r0

        # ** patch 
        self.patch_vert_ids = {}

    @property
    def radius(self):
        return self._radius

    @property
    def bond_type(self):
        return self._bond_type

    @property
    def center(self):
        return self._center

    def _id_to_str(self, id) -> str:
        return chr(65+id)

    def _str_to_id(self, typename:str) -> int:
        return ord(typename)-65

    def _extract_patch_vertex_ids(self, angle:float, axis=1) -> np.ndarray:
        if angle < 0.0 or angle > 360.0:
            raise ValueError("keyword `angle` must be in range between 0 and 360")
        if axis not in (0,1,2):
            raise ValueError("keyword `axis` must be 0, 1 or 2")

        _, patch_points = np.where([self.verts[:,axis] - self.center[axis] >= self.radius*np.cos(np.deg2rad(angle))])

        return patch_points
    
    def assign_patch_surface(self, angle:float, axis=1, typeid=1):
        """
        assigns patch surface to sphere particle. 
        If called more than once, the previous result is overwritten.

        Parameter
        ---------
        angle (float) : open angle of patch measured from the axis `axis`.
        axis (float) : pole axis of the patch. 0=x, 1=y, z=2
        typeid (int) : type id of the vertex on the patch surface. it must be > 0 (= normal type).

        Note
        ----
        vertex type-ids are 

        """
        if typeid == 0:
            raise ValueError("Additional typeid must be > 0")
        _patch_vert_ids = self._extract_patch_vertex_ids(angle, axis)

        self.vert_types[_patch_vert_ids] = typeid
        self.patch_vert_ids[self._id_to_str(typeid)] = _patch_vert_ids
        self.append_new_type(self._id_to_str(typeid))


class FCCSphere:
    def __init__(self, radius:float, a=None, m=None) -> None:
        """
        create sphere with raidus `R` by curved out a fcc crystal structure (box length `L=2R`, lattice constant `a`)

        Parameters
        ------------
        radius : float
            radius of a sphere
        a : float
            lattice constant.
        m : float
            number of lattice in an axis.
        """
        self.radius = radius
        self.L = 2*radius
        if a and not m:
            rho = 4.0/a**3.0
        elif m and not a:
            rho = 4.0*m**3.0/self.L**3.0
        else:
            raise ValueError("Both `a` and `m` must not be specified")

        r = FCCSphere._make_fcc_pure(L=self.L, rho=rho)
        r_com = r.mean(axis=0)
        self.vert = r[np.linalg.norm(r - r_com, axis=1) < radius]
        self.vert -= r_com
        self.r_com = self.vert.mean(axis=0)
        self.Nvert = len(self.vert)
        self.vert_type = np.zeros(self.Nvert)

        # patch
        self.vert_type_kind = ["A"]

    def assign_patch_surface(self, angle:float, axis, type_id:int):
        """
        Assign atom_type to vertices consisting a patchy surface.

        Parameter
        ------------
        angle : float
            patch angle. measured from `axis`
        axis : float
            direction from which the patch angle is measured
        type_id : int
            atom_type id. It should be greater than 0.
        """
        if angle < 0.0 or angle > 360.0:
            raise ValueError("keyword `angle` must be in range between 0 and 360")
        if axis not in (0,1,2):
            raise ValueError("keyword `axis` must be 0, 1 or 2")
        if type_id == 0:
            raise ValueError("type_id 0 is assigned for normal atom_type")

        _, _ids = np.where(self.vert[:,axis] - self.r_com[:,axis]>= self.radius*np.cos(np.deg2rad(angle)))
        self.vert_type[_ids] = type_id
        _to_str = chr(65+type_id)
        self.vert_type_kind.append(_to_str)

    def delete_patch(self, type_id:int):
        _to_str = chr(65+type_id)
        if _to_str not in self.vert_type_kind:
            raise ValueError(f"Vertex kind {_to_str} (type_id = {type_id}) not found.")
        if type_id == 0:
            raise ValueError(f"Don't remove normal atom_type.")

        self.vert_type_kind.remove(_to_str)
        self.vert_type[self.vert_type == type_id] = 0

    def _get_lattice_number(L, rho):
        m = np.floor((L**3 * rho / 4.0)**(1.0 / 3.0))
        drho1 = np.abs(4.0 * m**3 / L**3 - rho)
        drho2 = np.abs(4.0 * (m + 1)**3 / L**3 - rho)
        if drho1 < drho2:
            return m
        else:
            return m + 1

    def _make_fcc_pure(L, rho):
        m = FCCSphere._get_lattice_number(L, rho)
        a = L / m
        ha = a * 0.5
        atoms = []
        for i in range(int(m**3)):
            ix = i % m
            iy = (i // m) % m
            iz = i // (m * m)
            x = ix * a
            y = iy * a
            z = iz * a
            atoms.append((x, y, z))
            atoms.append((x + ha, y + ha, z))
            atoms.append((x + ha, y, z + ha))
            atoms.append((x, y + ha, z + ha))
        return np.array(atoms) - L/2
    

def diamond_sphere(a, r, n):
    dc =np.array(
        [[0.0, 0.0, 0.0], # 0
         [0.0, 0.5, 0.5], # 1
         [0.5, 0.0, 0.5], # 2
         [0.5, 0.5, 0.0], # 3
         [0.25, 0.25, 0.25], # 4
         [0.25, 0.75, 0.75], # 5
         [0.75, 0.25, 0.75], # 6
         [0.75, 0.75, 0.25] # 7
        ]    
        )
    return create_crystal_sphere(dc, r, a, n)

def fcc_sphere(a, r, n):
    dc =np.array(
        [[0.0, 0.0, 0.0], # 0
         [0.0, 0.5, 0.5], # 1
         [0.5, 0.0, 0.5], # 2
         [0.5, 0.5, 0.0], # 3
        ]    
        )
    return create_crystal_sphere(dc, r, a, n)


def create_crystal_sphere(unit_lattice:np.ndarray, radius:float, lattice_constant:float, ncell:int):
    """create a sphere from a cube crystal."""
    n_atom_lattice = len(unit_lattice)
    pos = []
    for ix in range(1, ncell+1):
        for iy in range(1, ncell+1):
            for iz in range(1, ncell+1):
                for j in range(0, n_atom_lattice):
                    x = (unit_lattice[j][0]+(ix-1))*lattice_constant-0.5*ncell*lattice_constant
                    y = (unit_lattice[j][1]+(iy-1))*lattice_constant-0.5*ncell*lattice_constant
                    z = (unit_lattice[j][2]+(iz-1))*lattice_constant-0.5*ncell*lattice_constant
                    pos.append([x,y,z])
    
    # curving out the crystal to form a sphere
    pos = np.array(pos)
    pos_com = pos.mean(axis=0)
    pos_sphere = pos[np.linalg.norm(pos - pos_com, axis=1) < radius]    
    
    return pos_sphere
