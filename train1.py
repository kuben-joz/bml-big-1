import re
from collections import defaultdict

import numpy as np
from mpi4py import MPI
from numpy.dtypes import StringDType

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

fn = "amazon_reviews_2M.csv"
#fn = "test.csv"

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)

words = defaultdict(lambda: 0)
max_label = 0
with open(fn, "r") as f:
    for line in f:
        text, label = line_split_re.findall(line)[0]
        label = int(label)
        max_label = max(max_label, label)
        text = [str.lower(word) for word in word_filter_re.findall(text)]
        #for word in text:
        #    words[word] += 1

num_labels = np.array(max_label + 1, dtype=np.int32)


#def get_sum(d):
#    res = 0
#    for v in d.values():
#        res += v
#    return res


#counts = np.fromiter(words.values(), dtype=np.int32)
#words = np.fromiter(words.keys(), dtype=StringDType())
#print(len(words))

print(num_labels)



# np.fromstring
# np to array
# words = b",".join([*words])
# np.unique


# words, inc_scan = np.unique(words, return_counts=True)
# inc_scan = np.add.accumulate(inc_scan)
# exc_scan = np.roll(inc_scan, 1)
# exc_scan[0] = 0
# ind = np.empty((exc_scan.size + inc_scan.size,), dtype=exc_scan.dtype)
# ind[0::2] = exc_scan
# ind[1::2] = inc_scan
# del exc_scan
# del inc_scan
# np.reduceat(counts, ind)
# argsort(type='stable') # uses timsort with is good for nearly sorted arrays


# jsut do counts > 0 or return index
