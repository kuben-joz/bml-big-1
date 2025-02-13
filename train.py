import re
from collections import defaultdict

import numpy as np
from numpy.dtypes import StringDType
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

fn = "amazon_reviews_2M.csv"
#fn = "test.csv"

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)
# todo
# feature selection , mutual info https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
#
words = defaultdict(lambda: 0)
max_word = 0
with open(fn, "r") as f:
    for line in f:
        text, label = line_split_re.findall(line)[0]
        label = int(label)
        text = [str.encode(str.lower(word)) for word in word_filter_re.findall(text)]
        for word in text:
            max_word = max(max_word, len(word))
            words[word] += 1



counts = np.array([*words.values(), b''], dtype=np.int32)
words = np.array([*words.keys()])

ind = words.argsort()
words = words[ind]
counts = counts[ind]


# np.fromstring
# np to array
# words = b",".join([*words])
# np.unique



#words, inc_scan = np.unique(words, return_counts=True)
#inc_scan = np.add.accumulate(inc_scan)
#exc_scan = np.roll(inc_scan, 1)
#exc_scan[0] = 0
#ind = np.empty((exc_scan.size + inc_scan.size,), dtype=exc_scan.dtype)
#ind[0::2] = exc_scan
#ind[1::2] = inc_scan
#del exc_scan
#del inc_scan
# np.reduceat(counts, ind)
# argsort(type='stable') # uses timsort with is good for nearly sorted arrays