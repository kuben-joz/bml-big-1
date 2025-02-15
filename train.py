import re
from collections import defaultdict

import numpy as np
from mpi4py import MPI
from numpy.dtypes import StringDType
from scipy.sparse import csc_array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

fn = "amazon_reviews_2M.csv"
# fn = "test.csv"

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)

words_to_counts = defaultdict(lambda: defaultdict(lambda: 0))
labels = defaultdict(lambda: 0)
max_label = 0
with open(fn, "r") as f:
    for line in f:
        text, label = line_split_re.findall(line)[0]
        label = int(label)
        max_label = max(max_label, label)
        text = [str.lower(word) for word in word_filter_re.findall(text)]
        labels[label] += 1
        for word in text:
            words_to_counts[word][label] += 1

num_labels = np.array(max_label + 1, dtype=np.int32)
num_labels_req = comm.Iallreduce(MPI.IN_PLACE, num_labels, MPI.MAX)


def get_sum(d):
    res = 0
    for v in d.values():
        res += v
    return res > 1


counts = np.fromiter((get_sum(v) for v in words_to_counts.values()), dtype=np.int8)
words = np.fromiter(words_to_counts.keys(), dtype=StringDType())

ind = words.argsort()
words = words[ind]
words_orig = words.copy()
counts = counts[ind]

words_recv_buf = bytearray(256)

str_cnt_out = np.zeros(2, dtype=np.int32)
str_cnt_in = np.zeros_like(str_cnt_out)

i = 1
while i < world_size:
    words_send_buf = ",".join(words).encode()
    str_cnt_out[:] = [len(words_send_buf), counts.shape[0]]
    send_reqs = []
    recv_reqs = []
    recv_from = (rank - i) % world_size
    send_to = (rank + i) % world_size
    send_reqs.append(comm.Isend(str_cnt_out, send_to))
    send_reqs.append(comm.Isend(words_send_buf, send_to))
    send_reqs.append(comm.Isend(counts, send_to))

    comm.Recv(str_cnt_in, recv_from)
    new_buf_len = len(words_recv_buf)
    while new_buf_len < str_cnt_in[0]:
        new_buf_len *= 2
    words_recv_buf[len(words_recv_buf) :] = b"\0" * (new_buf_len - len(words_recv_buf))
    recv_reqs.append(comm.Irecv(words_recv_buf, recv_from))
    counts_recv_buf = np.zeros(str_cnt_in[1], dtype=np.int8)
    recv_reqs.append(comm.Irecv(counts_recv_buf, recv_from))

    recv_reqs[0].Wait()
    words = words.append(words_recv_buf[: str_cnt_in[0]].decode().split(","))

    words, ind, counts_new = np.unique(words, return_index=True, return_counts=True)
    counts_new = counts_new[counts_new > 1]
    recv_reqs[1].Wait()
    counts = counts.append(counts_recv_buf)
    counts = counts[ind]
    counts[counts_new] = 1
    MPI.Request.Waitall(send_reqs)
    i *= 2


num_labels_req.wait()

res = np.zeros((num_labels, words.shape[0]))

words = words[np.nonzero(counts)]

word_locs = np.isin(words, words_orig).nonzero()

for loc, word in zip(word_locs, words[word_locs]):
    for label, count in words_to_counts[word]:
        res[label][loc] = count




