import re
import sys
import time
from collections import defaultdict

import numpy as np
from numpy.dtypes import StringDType
from scipy.sparse import csr_array

rank = 0
world_size = 0


# f_in = f"data/{rank}.csv"
# f_out = f"out/1-{rank}.out"
f_in = "data/0.csv"
f_out = "out/test.csv"
if len(sys.argv) == 3:
    f_in = sys.argv[1]
    f_out = sys.argv[2]

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)

words_to_counts = defaultdict(lambda: [0, defaultdict(lambda: 0)])
labels = defaultdict(lambda: 0)
max_label = 0
with open(f_in, "r") as f:
    for line in f:
        text, label = line_split_re.findall(line)[0]
        label = int(label)
        max_label = max(max_label, label)
        text, counts = np.unique(
            np.fromiter(
                (str.lower(word) for word in word_filter_re.findall(text)),
                dtype=StringDType(),
            ),
            return_counts=True,
        )
        labels[label] += 1
        for word, count in zip(text, counts):
            cur_el = words_to_counts[word]
            cur_el[0] += count
            cur_el[1][label] += 1


if rank == 0:
    print("finished reading file")


mod = 8
bit_type = np.uint8
bit_len = (world_size + 1 + 7) // 8
dict_len = len(words_to_counts)
rank_bit = np.zeros(bit_len, dtype=bit_type)
rank_bit[rank // mod] = 1 << (rank % mod)
world_bit = rank_bit.copy()
world_bit[world_size // mod] |= 1 << (world_size % mod)


num_labels = np.array(max_label + 1, dtype=np.int32)


counts = np.fromiter(
    (world_bit if v[0] > 1 else rank_bit for v in words_to_counts.values()),
    dtype=np.dtype((bit_type, bit_len)),
    count=dict_len,
)

words = np.fromiter(words_to_counts.keys(), dtype=StringDType(), count=dict_len)

# mpi --------------------------------------------------------------------------------
ind = words.argsort()
words = words[ind]
words_orig = words.copy()
counts = counts[ind]

if rank == 0:
    print("finished exchange of strings")

# final ---------------------------------------------------------

counts = np.bitwise_count(counts)

# this should upcast automatically

counts = np.add.reduce(counts, axis=1) > 1

words = words[counts]


label_counts = np.zeros(num_labels, dtype=np.int32)
for label, count in labels.items():
    label_counts[label] = count


res = np.zeros((words.shape[0], num_labels), dtype=np.int32)
word_locs = np.isin(words, words_orig).nonzero()[0]

for loc, word in zip(word_locs, words[word_locs]):
    for label, count in words_to_counts[word][1].items():
        res[loc][label] = count

res = np.transpose(res)

res = res.astype(np.double)

label_counts = label_counts.astype(np.double)[:, np.newaxis]
label_zeros = label_counts == 0
label_counts[label_zeros] = 1.0

res /= label_counts

label_counts[label_zeros] = 0.0
label_sum = label_counts.sum()


label_counts /= label_sum


res = np.concatenate((res, label_counts), axis=1)

with open(f_out, "w") as f:
    f.write(",".join(words))
    f.write("\n")
    np.savetxt(f, res, delimiter=",")