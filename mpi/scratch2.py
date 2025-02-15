import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


dat = np.zeros(1, dtype=np.int32)

dat[0] = rank

if rank == 0:
    comm.Reduce(None, dat, op=MPI.SUM, root=0)
    print(dat)
else:
    comm.Reduce(dat, None, op=MPI.SUM, root=0)
