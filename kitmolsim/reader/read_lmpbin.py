from kitmolsim.reader import read_lammps as rlmp
import numpy as np
import struct
import os
from typing import Union, List, Iterator

BIGINT = 8
INT = 4
DOUBLE = 8
FLOAT = 4

_LAMMPS_KEYS = {
    "id": ("ids", None),
    "type": ("atype", None),
    "x": ("pos", 0),
    "y": ("pos", 1),
    "z": ("pos", 2),
    "xu": ("pos", 0),
    "yu": ("pos", 1),
    "zu": ("pos", 2),
    "xs": ("pos", 0),
    "ys": ("pos", 1),
    "zs": ("pos", 2),
    "xsu": ("pos", 0),
    "ysu": ("pos", 1),
    "zsu": ("pos", 2),
    "vx": ("vel", 0),
    "vy": ("vel", 1),
    "vz": ("vel", 2),
    "ix": ("img", 0),
    "iy": ("img", 1),
    "iz": ("img", 2),
}

class LAMMPSBINReader:
    """
    A file reader for LAMMPS binary dump file (.lammpsbin).

    This reader suppors indexing and slicing.
    
    This reader supports per-atom attributes listed below:
    - atom id (`id`)
    - atom type (`type`)
    - wrapped coordinates (`x` `y` `z`)
    - unwrapped coordinates (`xu` `yu` `zu`)
    - scaled coordinates (`xs` `ys` `zs`)
    - unwrapped and scaled coordinates (`xsu` `ysu` `zsu`)
    - image flags (`ix` `iy` `iz`)
    - velocity (`vx` `vy` `vz`)

    Any attributes not listed above can also be read, but not stored in a frame object.

    Note
    ---------

    Coordinates, velocity, and image flags must contain all 3-components for each atom.

    If the dump file has at least 2 set of coordinates (e.g. wrapped coordinates and unwrapped coordinates), 
    This reader stores only attributes detected at last.
    
    This reader assumes constant atom number in each frame stored in a file. 

    """
    def __init__(self, filename:str) -> None:
        """
        Initialize LAMMPSBINReader object with the given LAMMPS dump file.

        Args:
            filename (str): Path to the LAMMPS dump file.
        """
        _fname = os.path.expanduser(filename)
        self._f = open(_fname, mode="rb")
        self._fsize = os.path.getsize(_fname)
        self.__read_1st_frame()

    def __read_1st_frame(self):
        """
        read the 1st frame and get the size of a frame and number of frames
        """
        self._f.seek(0)
        self._frame_size, frame0 = self.read_frame()
        self._f.seek(0)
        self._nframe = self._fsize//self._frame_size
        self._loc = [n*self._frame_size for n in range(self._nframe)] #location of a file for frames
    
    def __getitem__(self, key) -> Union[rlmp.LAMMPSTrajectoryFrame, List[rlmp.LAMMPSTrajectoryFrame]]:
        """
        Support indexing and slicing.

        Args:
            key (int or slice): Index or slice object.

        Returns:
            LAMMPSTrajectoryFrame or List[LAMMPSTrajectoryFrame]: The corresponding frame(s).
        """
        if isinstance(key, int):
            return self.read_at(key)
        elif isinstance(key, slice):
            start = key.start if key.start else 0
            stop  = key.stop  if key.stop else self._nframe
            step  = key.step  if key.step else 1
            return [self.read_at(n) for n in range(start, stop, step)]
        else:
            raise TypeError("Index must be an int or slice.")

    def __iter__(self) -> Iterator[rlmp.LAMMPSTrajectoryFrame]:
        """
        Allow iteration over the frames.

        Yields:
            LAMMPSTrajectoryFrame: Next frame.
        """
        for n in range(self._nframe):
            yield self.read_at(n)

    def __len__(self) -> int:
        return self._nframe

    def read_at(self, frame:int) -> rlmp.LAMMPSTrajectoryFrame:
        """
        Read a single trajectory frame from the given number of frame.
        """
        if frame > self._nframe-1:
            raise ValueError("index out of range")
        self._f.seek(self._loc[frame])
        _, Frame = self.read_frame()
        return Frame
    
    def read_frame(self):
        """
        read a frame from the file
        
        returns:
           loc (int): current file position
           frame (LAMMPSTrajectoryFrame): a trajectory frame

        """
        ntimestep_bytes = self._f.read(BIGINT)
        if not ntimestep_bytes:
            return
        
        ntimestep = struct.unpack('q', ntimestep_bytes)[0]#; print(ntimestep)  # 'q' is for int64 (bigint)
        if ntimestep < 0:
            magic_len = -ntimestep
            magic_str = self._f.read(magic_len).decode() #; print(f"magis str = {magic_str}")
            endian    = struct.unpack("i", self._f.read(INT))[0] #; print(endian)
            revision  = struct.unpack("i", self._f.read(INT))[0] #; print(f"rev = {revision}")
            ntimestep = struct.unpack("q", self._f.read(BIGINT))[0] #; print(f"time = {ntimestep}")

        # Example: Read the number of atoms (bigint)
        natoms    = struct.unpack('q', self._f.read(BIGINT))[0] #; print(f"Number of Atoms: {natoms}")
        triclinic = struct.unpack('i', self._f.read(INT))[0] #; print(triclinic)
        boundary  = struct.unpack("i"*6, self._f.read(INT*6)) #; print(boundary)

        # Read bounding box data (assuming 3 double pairs)
        xlo, xhi = struct.unpack('d'*2, self._f.read(BIGINT*2))
        ylo, yhi = struct.unpack('d'*2, self._f.read(BIGINT*2))
        zlo, zhi = struct.unpack('d'*2, self._f.read(BIGINT*2))
        _box = np.array([xhi-xlo, yhi-ylo, zhi-zlo])#; print(_box)
        sizeone  = struct.unpack('i', self._f.read(INT))[0] #; print(f"{sizeone} columne per line")
        if magic_str and revision > 1:
            # unit
            len_     = struct.unpack('i', self._f.read(INT))[0] #; print(len_)
            if len_ > 0:
                # UNIT: 
                pass
            # TIME
            flag     = b'\x00'
            flag     = self._f.read(1)
            if flag and flag != b"\x00":
                time = struct.unpack("d", self._f.read(DOUBLE))[0] #; print(f"time = {time}")
            # column
            len_ = struct.unpack('i', self._f.read(INT))[0] #; print(f"collen = {len_}")
            columuns = self._f.read(len_).decode() #; print(f"columns = {columuns} ({len(columuns)})")
        nchunk  = struct.unpack('i', self._f.read(INT))[0] #; print(f"written by {nchunk} proccessors")

        # read data buffer
        _natom_per_chunk = int(np.rint(natoms/nchunk))+10 #HACK: natoms/nchunkだと足りないことがあるので, 多めに(10個だけ)取っておく. 
        _data_buffer = np.empty((nchunk,sizeone,_natom_per_chunk), dtype=np.float64)
        _true_counts = np.empty(nchunk, dtype=np.int32)
        for ichunk in range(nchunk):
            n  = struct.unpack('i', self._f.read(INT))[0] #; print(f"N={n}")
            _true_counts[ichunk] = n//sizeone
            _data_buffer[ichunk,:,:n//sizeone] = np.frombuffer(self._f.read(DOUBLE*n), count=n).reshape([-1,sizeone]).T

        # create attributes from buffered data
        _offsets = np.append([0], np.cumsum(_true_counts))
        colnames = columuns.split(" ") #e.g. id type x y z

        # map from lammps dump keys to read_lammps.LAMMPSTrajectoryFrame keys
        # The original idea is came from https://github.com/mphowardlab/lammpsio/blob/main/src/lammpsio/dump.py
        _schema = {}
        for i, colname in enumerate(colnames):
            try:
                key, keyid = _LAMMPS_KEYS[colname]
                if keyid is None:
                    # for single component (id, type)
                    _schema[key] = i
                else:
                    # for 3-component vector (pos, vel, img)
                    if key not in _schema: _schema[key] = [None, None, None]
                    _schema[key][keyid] = i
                
            except KeyError:
                raise KeyError(f"read_lmpbin does not know such field name '{colname}'")

        # validate schema
        for key in ("pos", "vel", "img"):
            if key in _schema and any(x is None for x in _schema[key]):
                raise IOError("read_lmpbin requires 3-component vectors for 'pos', 'vel', and 'img'.")
            
        # get data from buffer
        _vel = None #optional field valriable
        _img = None #optional field valriable
        for key in _schema:
            if key == "ids":
                # _ids = _data_buffer[:,nc,:]
                nc = _schema[key]
                _ids = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)

            elif key == "atype":
                nc = _schema[key]
                _atype = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)
        
            elif key == "pos":
                ncx, ncy, ncz = _schema[key]
                _pos_x = self._getarr(natoms, nchunk, _data_buffer[:,ncx,:], _offsets, _true_counts)
                _pos_y = self._getarr(natoms, nchunk, _data_buffer[:,ncy,:], _offsets, _true_counts)
                _pos_z = self._getarr(natoms, nchunk, _data_buffer[:,ncz,:], _offsets, _true_counts)
                _pos = np.column_stack((_pos_x, _pos_y, _pos_z))
        
            elif key == "vel":
                ncx, ncy, ncz = _schema[key]
                _vel_x = self._getarr(natoms, nchunk, _data_buffer[:,ncx,:], _offsets, _true_counts)
                _vel_y = self._getarr(natoms, nchunk, _data_buffer[:,ncy,:], _offsets, _true_counts)
                _vel_z = self._getarr(natoms, nchunk, _data_buffer[:,ncz,:], _offsets, _true_counts)
                _vel = np.column_stack((_vel_x, _vel_y, _vel_z))

            elif key == "img":
                ncx, ncy, ncz = _schema[key]
                _img_x = self._getarr(natoms, nchunk, _data_buffer[:,ncx,:], _offsets, _true_counts)
                _img_y = self._getarr(natoms, nchunk, _data_buffer[:,ncy,:], _offsets, _true_counts)
                _img_z = self._getarr(natoms, nchunk, _data_buffer[:,ncz,:], _offsets, _true_counts)
                _img = np.column_stack((_img_x, _img_y, _img_z))

        loc = self._f.tell()

        frame = rlmp.LAMMPSTrajectoryFrame(ntimestep, natoms, _ids, _atype, _pos, _box, 
                                           vel=_vel, img=_img)        
        return loc, frame
        

    def _getarr(self, natom, nchunk:int, buffer:np.ndarray, offsets:np.ndarray, true_counts:np.ndarray) -> np.ndarray:
        retval = np.empty(natom)
        for n in range(nchunk):   
            st = offsets[n]
            en = offsets[n+1]
            retval[st:en] = buffer[n,:true_counts[n]]
        return retval

if __name__ == "__main__":
    fname = "~/annealing.lammpsbin"
    # fname = "pos.bin"
    traj = LAMMPSBINReader(fname) #contains 11 frames (0~10)
    print(traj[1].natom)
    print(traj[1].timestep)
    print(traj[1].pos.shape)
    print(traj[1].vel)
    # print(traj[0].timestep)
    # print(traj[10].timestep)
    # # print(traj[11].timestep)
    # for frame in traj:
    #     print(frame.timestep)
    # for frame in traj[0:10:2]: #(start:end:step) where end <= number of frames
    #     print(frame.timestep)