from kitmolsim.reader import read_lammps as rlmp
import numpy as np
import struct
import os
from typing import Union, List, Iterator

BIGINT = 8
INT = 4
DOUBLE = 8
FLOAT = 4
class LAMMPSBINReader:
    """
    A file reader for LAMMPS binary dump file

    Note
    ---------
    valid only for `ITEM: ATOMS id type xu yu zu`
    """
    def __init__(self, filename:str) -> None:
        """
        Initialize LAMMPSBINReader object with the given LAMMPS dump file.

        Args:
            filename (str): Path to the LAMMPS dump file."""
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
        
        Note
        ---------
        valid only for `ITEM: ATOMS id type xu yu zu`
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
        colname = columuns.split(" ")
        for nc, col in enumerate(colname):
            if col == "id":
                # _ids = _data_buffer[:,nc,:]
                _ids = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)
            elif col == "type":
                _atype = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)
            elif col == "xu":
                _pos_x = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)
            elif col == "yu":
                _pos_y = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)
            elif col == "zu":
                _pos_z = self._getarr(natoms, nchunk, _data_buffer[:,nc,:], _offsets, _true_counts)
            else:
                raise ValueError(f"unknown keyword {col} in column")

        #NOTE: x,y,z座標全てある前提. どれか欠けるとエラー. 
        _pos = np.column_stack((_pos_x, _pos_y, _pos_z))

        loc = self._f.tell()

        frame = rlmp.LAMMPSTrajectoryFrame(ntimestep, natoms, _ids, _atype, _pos, _box)        
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
    print(traj[1].timestep)
    # print(traj[0].timestep)
    # print(traj[10].timestep)
    # # print(traj[11].timestep)
    # for frame in traj:
    #     print(frame.timestep)
    # for frame in traj[0:10:2]: #(start:end:step) where end <= number of frames
    #     print(frame.timestep)