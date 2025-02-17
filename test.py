import re
from collections import defaultdict

import numpy as np
from mpi4py import MPI
from numpy.dtypes import StringDType
from scipy.sparse import csr_array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


arr_in = np.arange(4, dtype=np.int32)
arr_out = np.zeros(100, dtype=np.int32)

arr_in += rank

def myreduce(xmem, ymem, dt):
  x = np.frombuffer(xmem, dtype=np.int32)
  y = np.frombuffer(ymem, dtype=np.int32)

  z = x + y
  print(f"x:{x.shape} y:{y.shape}")
  y[:] = z


op = MPI.Op.Create(myreduce, commute=True)

comm.Reduce(arr_in, arr_out, op=op)
