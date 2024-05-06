import numpy as np
import collections.abc
import typing

class LAMMPSTrajectoryFrame:
    """
    A snapshot of trajectory.
    This contains:
    - timestep: timestep
    - natom: number of atoms
    - atype: atom types ((natom,) ndarray)
    - ids: atom ids ((natom,) ndarray)
    - pos: atom positions ((natom,3) ndarray)
    - box: system box dimensions ((6,) ndarray)
    """
    def __init__(self, timestep:int, natom:int, 
                 ids:np.ndarray, 
                 atype:np.ndarray, 
                 pos:np.ndarray, 
                 box:np.ndarray,
                 vel=np.ndarray,
                 img=np.ndarray):
        self.timestep: int = timestep
        self.natom: int = natom
        self.atype: np.ndarray = atype
        self.ids: np.ndarray = ids
        self.pos: np.ndarray = pos
        self.box: np.ndarray = box
        self.vel = None
        self.img = None
    

class LAMMPSTrajectory(collections.abc.Container):
    """
    LAMMPS trajectory containing multiple frames.
    """

    def __init__(self):
        self.frames: typing.List[LAMMPSTrajectoryFrame] = []
        
    def append(self, frame:LAMMPSTrajectoryFrame):
        if not isinstance(frame, LAMMPSTrajectoryFrame):
            raise TypeError("frame must be LAMMPSTrajectoryFrame")
        self.frames.append(frame)

    def __iter__(self):
        yield from self.frames

    def __len__(self):
        return len(self.frames)
    
    def __contains__(self, __x: object) -> bool:
        return __x in self.frames
    
    def __getitem__(self, key):
        return self.frames[key]
    
    def unwrap(self) -> None:
        """ 
        unwraps all coordinates contained in frames.

        All positions except the initial position are to be unwrapped.
        """
        _nframe = len(self.frames)
        if _nframe <= 1:
            raise ValueError(f"trajectory does not have enough frames for coordinates to be unwrapped.")
        
        _natom = self.frames[0].natom
        for i in range(_nframe-1):
            _unwrap_pos(_natom, 
                        prev_pos=self.frames[i].pos, 
                        next_pos=self.frames[i+1].pos,
                        box=self.frames[i].box)
            
    def wrap(self):
        """ wrap the unwrapped coordinates"""
        for snap in self.frames:
            L = snap.box
            dL = L/2
            snap.pos = (snap.pos + dL)%L - dL


def _unwrap_pos(natom, prev_pos:np.ndarray, next_pos:np.ndarray, box:np.ndarray):
    """ 
    unwrao corrdinates from the previous position.
    """
    dL = box/2
    for i in range(natom):
        dx = next_pos[i,:] - prev_pos[i,:]
        if dx[0] > dL[0]:
            next_pos[i,0] -= ( dx[0]+dL[0])//box[0]*box[0]
        if dx[0] < -dL[0]:
            next_pos[i,0] += (-dx[0]+dL[0])//box[0]*box[0]
    

def read_dumpfile(filename:str):
    """
    read dump file written by the command "dump" with the keyword option "atom". 

    Parameter
    ----------
    filename: str
        filename (.lammpstrj)
    """
    trajectory = LAMMPSTrajectory()
    n = 0
    with open(filename, mode="r") as f:

        keyword = f.readline() #read first line
        while (keyword):
            if "ITEM: TIMESTEP" in keyword:
                timestep = int(f.readline().split(" ")[0])

            if "ITEM: NUMBER OF ATOMS" in keyword:
                natom = int(f.readline().split(" ")[0])
                
            if "ITEM: BOX BOUNDS" in keyword:
                lxo, lxh = [float(item) for item in f.readline().split()]
                lyo, lyh = [float(item) for item in f.readline().split()]
                lzo, lzh = [float(item) for item in f.readline().split()]
                box = np.array([lxh-lxo, lyh-lyo, lzh-lzo])

            if "ITEM: ATOMS" in keyword:
                # count number of items (e.g. "ITEM: ATOMS type id x y z ...")
                nitem = len(keyword.split(" ")[2:])
                # For simplicity, assume the format is "atom" and "custom" with 5 tags (type, id, coordinates (x,y,z or xs,ys,zs or xu,yu,zu))
                assert(nitem == 5)
                buffer = np.loadtxt(f, max_rows=natom)
                
                _frame = LAMMPSTrajectoryFrame(timestep, natom, buffer[:,0], buffer[:,1], buffer[:,2:5], box)

                trajectory.append(_frame)
                print(f"\r Number of item = {n}", end="")
                n += 1

            keyword = f.readline() # read the first line of next frame
    print("")
    return trajectory
    
if __name__ == "__main__":
    import argparse
    import os
    description = """
    
    read LAMMPS trajectory (.lammpstrj) and dump into .txt or binary data

    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("path_to_file")
    args = parser.parse_args()

    path_to_file = args.path_to_file
    savedir, _ = os.path.split(path_to_file)
    
    trj = read_dumpfile(path_to_file)
    ntot = len(trj)
    print(f"num of frame: {ntot}")

    for i in range(3):
        print(trj.frames[i].pos.max())

    trj.wrap()

    for i in range(3):
        print(trj.frames[i].pos.max())




    