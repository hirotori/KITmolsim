"""
icomesh
---------

creating icosahedron, icosphere and spherocylinder

Made with chatGPT3

References
-----------

https://suzulang.com/cpp-code-ico-q-1/

http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html

"""
import numpy as np


def create_icosahedron():
    # 正20面体の頂点座標を定義
    t = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = np.array([
        # rectangle 1
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        # rectangle 2
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        # rectangle 3
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1]
    ], dtype=float)

    # 正20面体の面を構成する頂点のインデックス
    faces = np.array([
        # 5 faces around point 0
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        # 5 faces adjacent to above 5 faces
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        # 5 faces around point 3
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        # 5 faces adjacent to above 5 faces
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ], dtype=np.int32)

    return vertices, faces

class Icosphere:
    """ Icosahedron object. """
    def __init__(self, radius:float, center=[0,0,0], nsub=2) -> None:
        """
        create Icosahedron.

        Parameter
        ---------
        radius (float) : radius of the icosphere
        center (ndarray) : position of center-of-mass. [0,0,0] by default
        nsub (int) : number of subdivision from icoshaedron.

        """
        if radius <= 0:
            raise ValueError("radius must be > 0")

        if len(center) != 3:
            raise ValueError("center must be an array with 3 components (x,y,z)")
        
        if nsub <= 0:
            raise ValueError("nsub must be > 0")
    
        
        self.R = radius
        self.center = center
        self.verts, self._faces = create_icosphere(radius=radius, num_subdivisions=nsub)
        self.verts = self.verts + np.array(center)
        self.nvert = len(self.verts)
        self.pt_neighbors = create_adjacency_list(self.faces)

    @property
    def vertices(self):
        """vertices of icosphere"""
        return self.verts
    
    @property
    def faces(self):
        """faces of icosphere"""
        return self._faces

    @property
    def point_neighbors(self):
        """list of nearest neighbors to each vertices"""
        return self.pt_neighbors
    

def _normalize(v, axis=None):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def _calc_midpoint(edge:set, searched_edge_list:list, midpoint_ids:list, vi, vj, new_verts):
    """ compute midpoint of edge and return the id of midpoint"""
    try:
        edge_id = searched_edge_list.index(edge)
        midp_id = midpoint_ids[edge_id]
    except:
        vij = _normalize((vi + vj) / 2.0) #edge 0
        new_verts = np.vstack([new_verts, vij])
        midp_id = len(new_verts) - 1
        searched_edge_list.append(edge)
        midpoint_ids.append(midp_id) 

    return  midp_id, new_verts


def _subdivide(vertices, faces, radius):
    """
    Subdivides given icosahedron and project any particle onto surface of a sphere with radius r.
    """
    new_vertices = vertices.copy()
    new_faces = []
    searched_list = []
    midpoint_idx = []
    for face in faces:
        # get vertices from a face
        iv1, iv2, iv3 = face[:]
        v1 = vertices[iv1]
        v2 = vertices[iv2]
        v3 = vertices[iv3]

        edge0 = {iv1,iv2}
        edge1 = {iv2,iv3}
        edge2 = {iv3,iv1}

        # Append new mid-points on a edge and normalize by radius R if the edge has not been searched.
        v12_index, new_vertices = _calc_midpoint(edge0, searched_list, midpoint_idx, v1, v2, new_vertices)
        v23_index, new_vertices = _calc_midpoint(edge1, searched_list, midpoint_idx, v2, v3, new_vertices)
        v31_index, new_vertices = _calc_midpoint(edge2, searched_list, midpoint_idx, v3, v1, new_vertices)

        # Append new faces
        new_faces.append([iv1      , v12_index, v31_index])
        new_faces.append([v12_index, iv2      , v23_index])
        new_faces.append([v31_index, v23_index, iv3      ])
        new_faces.append([v12_index, v23_index, v31_index])

    return new_vertices, new_faces

def create_icosphere(radius, num_subdivisions):
    """creates icosphere."""
    # create icosahedron
    vertices, faces = create_icosahedron()
    # normalize
    # 半径1の単位球面に点を移動
    v_norm = np.linalg.norm(vertices, axis=1)
    vertices = vertices/v_norm.reshape([-1,1])

    for _ in range(num_subdivisions):
        vertices, faces = _subdivide(vertices, faces, radius)

    return vertices*radius, faces

def create_adjacency_list(faces):
    adjacency_list = {}

    for face in faces:
        for vertex in face:
            if vertex not in adjacency_list:
                adjacency_list[vertex] = []

    for face in faces:
        v1, v2, v3 = face
        adjacency_list[v1].extend([v2, v3])
        adjacency_list[v2].extend([v3, v1])
        adjacency_list[v3].extend([v1, v2])

    # Remove duplicates and sort the lists
    for vertex in adjacency_list:
        adjacency_list[vertex] = sorted(list(set(adjacency_list[vertex])))

    return adjacency_list


if __name__ == "__main__":
    # Generate an icosphere with radius R=3 (nsub = 2 for example)
    import pyvista as pv
    radius = 3.0
    num_subdivisions = 2
    r_icos, faces = create_icosphere(radius, num_subdivisions)
    print(r_icos.shape)
    icosphere = pv.Icosphere(radius=radius, nsub=num_subdivisions)
    # test plot
    print(icosphere.points.shape)
    plotter = pv.Plotter()
    plotter.add_points(r_icos, point_size=23, render_points_as_spheres=True)
    plotter.add_mesh(icosphere)
    plotter.show()
    adj_list = create_adjacency_list(faces)
    for n in range(len(adj_list)):
        print(adj_list[n])