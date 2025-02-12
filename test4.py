import re
from collections import defaultdict

import numpy as np
from numpy.dtypes import StringDType

import torch

torch.set_num_threads(1)

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)



arr = torch.from_numpy(np.load('amazon.npy', allow_pickle=True))

arr, _ = arr.sort()
