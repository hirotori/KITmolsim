from . import util
import numpy as np

class RectangleParticle:
    """
    A rectangle object composed of a set of vertices.

    This creates a rectangle object composed of vertices arranged in square lattice on each surface.

    """
    def __init__(self, nx:int, ny:int, nz:int, scale=1.0, center_bond=True) -> None:
        """
        Creates a rectangle object. This object consists of `nx*ny*nz` vertices. 

        The vertices are arranged in a square lattice on surfaces of an object.
        Each distance between nearest neighbors is `1.0*scale`. 
        This object also has "bonding" information; i.e. This object connects the vertices with their nearest neighbors. 
        if `center_bond` is true, The diametrically opposite pairs are also connected with each other.

        Parameter
        ---------
        nx (int) : number of vertices along the x-direction of a square lattice
        ny (int) : number of vertices along the y-direction of a square lattice
        nz (int) : number of vertices along the z-direction of a square lattice
        scale (float) : the distance between nearest neighbor vertices. 1.0 as default
        center_bond (bool) : connects diametrically opposite vertices if true

        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Nvert = None
        self.verts = None
        self.vert_type = None
        self.bonds = None
        self.bond_r0 = None
        self.bond_types = None

        self.verts, self.bonds, self.bond_r0, self.bond_types = rectangle_particle(nx, ny, nz, center_bond=center_bond)
        self.Nvert = len(self.verts)
        self.vert_type = np.zeros(self.Nvert, dtype=np.int32)
        self.vert_type_kinds = ["A"] # "A" is corresponding with vert_type=0
        self.bond_type_kinds = [f"bond{i}" for i in range(len(self.bond_r0))]
        self.L = ((nx-1)*scale, (ny-1)*scale, (nz-1)*scale)

        #scaling
        self.verts *= scale
        self.bond_r0 *= scale

        # surfaces
        # on XZ plane (b, t)
        self.patch_dict = {}
        self.patch_dict["XZ-"] = np.hstack(tuple(np.arange(1,nx-1)+j*nx for j in range(1,nz-1)))
        self.patch_dict["XZ+"] = self.patch_dict["XZ-"] + nx*nz
        offset = 2*nx*nz
        self.patch_dict["XY-"] = offset + np.hstack(tuple(np.arange(1,nx-1)+j*nx for j in range(ny-2)))
        self.patch_dict["XY+"] = self.patch_dict["XY-"] + nx*(ny-2)
        offset += 2*nx*(ny-2)
        self.patch_dict["YZ-"] = offset + np.hstack(tuple(np.arange(ny-2)+j*(ny-2) for j in range(nz-2)))  
        self.patch_dict["YZ+"] = self.patch_dict["YZ-"] + (ny-2)*(nz-2)

    def assign_vert_type(self, patch:str, type_id:int):

        int_to_alp = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G"}

        if patch not in self.patch_dict.keys():
            raise ValueError('`patch` must be: "XZ-", "XZ+", "XY-", "XY+", "YZ-" or "YZ+".')
        
        if type_id == 0:
            print("WARNING: type_id = 0 is a default type.")

        if type_id not in int_to_alp.keys():
            raise ValueError("type_id is too big. It must be from 0 to 6.")
        
        if int_to_alp[type_id] not in self.vert_type_kinds:
            self.vert_type_kinds.append(int_to_alp[type_id])
            self.vert_type_kinds.sort()

        self.vert_type[self.patch_dict[patch]] = type_id
                

def _face_pair(ix:int, jy:int, offset=0):
    """
    search pairs of vertices on a rectilinear plane surface
    """
    bond_h = np.column_stack((np.arange(ix-1),np.arange(ix-1)+1))
    bond_h = np.vstack(tuple(bond_h + j*ix for j in range(jy)))
    bond_v = np.column_stack((np.arange(ix),np.arange(ix)+ix))
    bond_v = np.vstack(tuple(bond_v + j*ix for j in range(jy-1)))
    return bond_h+offset, bond_v+offset

def cube_particle(n:int):
    """
    create a cube particle.
    This is an alias of `rectangle_particle`

    Parameter
    ------------
    n : int
        number of vertices in an edge.

    Returns
    -------------
    verts : 
        vertices.
    bonds :
        bonds
    r0 :
        distances between pairs in equilibrium
    bond_types :
        bond types.
    """
    return rectangle_particle(nx=n, ny=n, nz=n)

def rectangle_particle(nx:int, ny:int, nz:int, center_bond=True):
    """
    create a rectangle particle.

    Returns
    -------------
    verts : 
        vertices.
    bonds :
        bonds
    r0 :
        distances between pairs in equilibrium
    bond_types :
        bond types.

    Notes
    --------
    Edge lengths are Lx=(nx-1), Ly=(ny-1) and Lz=(nz-1).
    """
    # ************************
    # ****** XZ plane ********
    # ************************
    # + - + - + - + - + - +  <---- Lxline (upper)
    # |   |   |   |   |   |
    # + - + - + - + - + - +
    # |   |   |   |   |   |
    # + - + - + - + - + - +  
    # |   |   |   |   |   |
    # + - + - + - + - + - +
    # |   |   |   |   |   |
    # + - + - + - + - + - +  
    # |   |   |   |   |   |
    # + - + - + - + - + - +  <---- Lxline (lower)
    # ↑                   ↑
    # Lzline (left)       Lzline (right)
    #
    # z
    # ^
    # |
    # + -- > x

    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("nx, ny and nz must be > 3")

    seed_x = np.arange(nx)
    seed_y = np.arange(ny)
    seed_z = np.arange(nz)
    x, z = np.meshgrid(seed_x, seed_z, indexing="xy")
    # XZ plane
    XZplane_b = np.column_stack((x.flatten(), np.zeros(x.size), z.flatten()))
    XZplane_t = XZplane_b.copy()
    XZplane_t[:,1] += ny - 1

    # ************************
    # ****** XY plane ********
    # ************************
    # + .. + . + . + . + .. +  <== XZplane_t (id = nx*nz ~ 2*nx*nz)
    # :    :   :   :   :    :
    # + -- + - + - + - + -- +
    # |    |   |   |   |    |
    # + -- + - + - + - + -- +  
    # |    |   |   |   |    |
    # + -- + - + - + - + -- +
    # |    |   |   |   |    |
    # + -- + - + - + - + -- +  
    # :    :   :   :   :    :
    # + .. + . + . + . + .. +  <== XZplane_b (id = 0 ~ nx*nz-1)
    #
    # y
    # ^
    # |
    # + -- > x
    x, y = np.meshgrid(seed_x, seed_y[1:ny-1], indexing="xy")
    XYplane_b = np.column_stack((x.flatten(), y.flatten(), np.zeros(x.size)))
    XYplane_t = XYplane_b.copy()
    XYplane_t[:,2] += nz - 1

    # ************************
    # ****** YZ plane ********
    # ************************
    # + .. + . + . + . + .. +  
    # :    :   :   :   :    :
    # + .. + - + - + - + .. +
    # :    |   |   |   |    :
    # + .. + - + - + - + .. +  
    # :    |   |   |   |    :
    # + .. + - + - + - + .. +
    # :    |   |   |   |    :
    # + .. + - + - + - + .. +  
    # :    :   :   :   :    : 
    # + .. + . + . + . + .. +  
    # ↑                     ↑
    # XZplane_t       XZplane_b
    #
    #        z
    #        ^
    #        |
    # y < -- + 
    y, z = np.meshgrid(seed_y[1:ny-1], seed_z[1:nz-1], indexing="xy")
    YZplane_b = np.column_stack((np.zeros(y.size), y.flatten(), z.flatten()))
    YZplane_t = YZplane_b.copy()
    YZplane_t[:,0] += nx - 1

    verts = np.vstack((XZplane_b, XZplane_t, XYplane_b, XYplane_t, YZplane_b, YZplane_t))

    # ********************
    # *** create pairs ***
    # ********************
    # num of vertices per plane
    numv = [nx*nz, nx*(ny-2), (ny-2)*(nz-2)]

    # make bond pair (surface)
    # ** XZ plane (bottom) ** 
    bond_h, bond_v = _face_pair(nx, ny)
    bondXZ = np.vstack((bond_h, bond_v))
    # ** XZ plane (top) ** 
    bondXZ = np.vstack((bondXZ, bondXZ.copy()+numv[0]))

    # ** XY plane ** 
    offset = numv[0]*2
    bond_h, bond_v = _face_pair(nx, ny-2, offset)
    bondXY_b = np.vstack((bond_h, bond_v))
    bondXY_t = bondXY_b.copy()+numv[1]
    # - pair between bottom of XZplane_b & XZplane_t 
    bond_v_bb = np.column_stack((offset+np.arange(nx), np.arange(nx)))
    bond_v_bt = np.column_stack((offset+np.arange(nx)+nx*(ny-3), np.arange(nx)+numv[0]))
    # - pair between top of XZplane_b & XZplane_t
    bond_v_tb = np.column_stack((offset+np.arange(nx)+numv[1], np.arange(nx)+(nz-1)*nx))
    bond_v_tt = np.column_stack((offset+np.arange(nx)+nx*(ny-3)+numv[1], np.arange(nx)+(nz-1)*nx+numv[0]))
    bondXY = np.vstack((bondXY_b, bond_v_bb, bond_v_bt, bondXY_t, bond_v_tb, bond_v_tt))

    # ** YZ plane **
    offset += numv[1]*2
    bond_h, bond_v = _face_pair(nx-2, ny-2, offset)
    bondYZ_b = np.vstack((bond_h, bond_v))
    bondYZ_t = bondYZ_b.copy()+numv[2]
    # - pair between left edge of XZplane_b & XZplane_t 
    bond_v_lxzb = np.column_stack((offset+np.arange(nz-2)*(ny-2), np.arange(1,nz-1)*nx))
    bond_v_lxzt = np.column_stack((offset+ny-3+np.arange(nz-2)*(ny-2), numv[0]+np.arange(1,nz-1)*nx))
    # - pair between left edge of XYplane_b & XYplane_t
    bond_v_lxyb = np.column_stack((offset+np.arange(ny-2), 2*numv[0]+np.arange(ny-2)*nx))
    bond_v_lxyt = np.column_stack((offset+np.arange(ny-2)+(nz-3)*(ny-2), 2*numv[0]+numv[1]+np.arange(ny-2)*nx))
    # - pair between right edge of XZplane_b & XZplane_t 
    bond_v_rxzb = np.column_stack((offset+numv[2]+np.arange(nz-2)*(ny-2), nx-1+np.arange(1,nz-1)*nx))
    bond_v_rxzt = np.column_stack((offset+numv[2]+ny-3+np.arange(nz-2)*(ny-2), numv[0]+nx-1+np.arange(1,nz-1)*nx))
    # - pair between right edge of XYplane_b & XYplane_t
    bond_v_rxyb = np.column_stack((offset+numv[2]+np.arange(ny-2), 2*numv[0]+ny-1+np.arange(ny-2)*nx))
    bond_v_rxyt = np.column_stack((offset+numv[2]+np.arange(ny-2)+(nz-3)*(ny-2), 2*numv[0]+numv[1]+nx-1+np.arange(ny-2)*nx))
    bondYZ = np.vstack((bondYZ_b, bond_v_lxzb, bond_v_lxzt, bond_v_lxyb, bond_v_lxyt, 
                        bondYZ_t, bond_v_rxzb, bond_v_rxzt, bond_v_rxyb, bond_v_rxyt))

    bonds = np.vstack((bondXZ, bondXY, bondYZ))
    nsurf_pair = len(bonds)

    # add bonds between diametrically opposed vertices
    if center_bond:
        opposed_bonds, opposed_bond_distances = util.search_opposed_pair(verts)
        bonds = np.vstack((bonds, opposed_bonds))

    # bond types, r0
    bond_types = np.zeros(len(bonds), dtype=np.int32) #surface bond type = 0
    r0 = np.array([1.0])

    # center bond
    if center_bond:
        r0 = np.append(r0, np.unique(opposed_bond_distances))
        for bond_typeId, _distance in enumerate(r0):
            group_ids = np.where(opposed_bond_distances == _distance)[0] #corresponding pair ids
            bond_types[nsurf_pair+group_ids] = bond_typeId

    return verts, bonds, r0, bond_types



class PackedRectangleParticle:
    def __init__(self, nx:int, ny:int, nz:int, scale = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Nvert = nx*ny*nz
        self.verts = None
        self.vert_type = np.zeros(self.Nvert)

        seed_x = np.arange(nx)
        seed_y = np.arange(ny)
        seed_z = np.arange(nz)
        _x, _y, _z = np.meshgrid(seed_x, seed_y, seed_z, indexing="ij")
        self.verts = np.column_stack((_x.flatten(), _y.flatten(), _z.flatten()))

        # adjacency
        self.bonds = []
        self._adj_mat = self._create_adjacency_matrix()
        for i in range(self._adj_mat.shape[0]):
            for j, flag in enumerate(self._adj_mat[i, i+1:]):
                if flag == 1: self.bonds.append([i, i+j+1])

        self.bonds = np.array(self.bonds)
        self.bond_r0 = np.full(len(self.bonds), fill_value=1.0*scale)
        self.bond_types = np.zeros(len(self.bonds))

    def _create_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.Nvert, self.Nvert), dtype=int)

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    current_point = self._indexing(i, j, k)

                    # Check neighboring points in x-direction
                    if i > 0:
                        adjacency_matrix[current_point, self._indexing(i-1, j, k)] = 1
                    if i < self.nx - 1:
                        adjacency_matrix[current_point, self._indexing(i+1, j, k)] = 1

                    # Check neighboring points in y-direction
                    if j > 0:
                        adjacency_matrix[current_point, self._indexing(i, j-1, k)] = 1
                    if j < self.ny - 1:
                        adjacency_matrix[current_point, self._indexing(i, j+1, k)] = 1

                    # Check neighboring points in z-direction
                    if k > 0:
                        adjacency_matrix[current_point, self._indexing(i, j, k-1)] = 1
                    if k < self.nz - 1:
                        adjacency_matrix[current_point, self._indexing(i, j, k+1)] = 1

        return adjacency_matrix


    def _indexing(self, i, j, k):
        """local indices to global id"""
        return k+i*self.nz+j*self.nz*self.nx

    def assign_vert_type(self):
        pass


if __name__ == "__main__":
    NX = NY = NZ = 5

    cube = RectangleParticle(NX, NY, NZ, center_bond=False)
    print(cube.Nvert)
    print(cube.patch_dict["XZ-"])
    print(cube.patch_dict["XZ+"])
    print(cube.patch_dict["XY-"])
    print(cube.patch_dict["XY+"])
    print(cube.patch_dict["YZ-"])
    print(cube.patch_dict["YZ+"])
    cube.assign_vert_type("XZ-", 1)
    cube.assign_vert_type("XY-", 1)
    print(cube.vert_type)

    pair_len = np.empty(len(cube.bonds))
    for n, ids in enumerate(cube.bonds):
        pair_len[n] = np.linalg.norm(cube.verts[ids[0]] - cube.verts[ids[1]])
    pair_lens, counts = np.unique(pair_len, return_counts=True)
    print(pair_lens)
    print(counts)

    packed_cube = PackedRectangleParticle(NX, NY, NZ)
    for j in range(NY):
        for i in range(NX):
            for k in range(NZ):
                id = packed_cube._indexing(i, j, k)
                print(packed_cube.verts[id,:], id)

    print(packed_cube._adj_mat)
    print(packed_cube.bonds)
    pair_len = np.empty(len(packed_cube.bonds))
    for n, ids in enumerate(packed_cube.bonds):
        pair_len[n] = np.linalg.norm(packed_cube.verts[ids[0]] - packed_cube.verts[ids[1]])
    pair_lens, counts = np.unique(pair_len, return_counts=True)
    print(pair_lens)
    print(counts)
