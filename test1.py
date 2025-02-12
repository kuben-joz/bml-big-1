import re
from collections import defaultdict

import numpy as np
from numpy.dtypes import StringDType

arr = np.load('amazon.npy', allow_pickle=True)

arr = np.unique(arr)