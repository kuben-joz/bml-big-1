import re
from collections import defaultdict

import numpy as np
from mpi4py import MPI
from numpy.dtypes import StringDType
from scipy.sparse import csr_array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

fn = f"data/{rank}.csv"
fn_out = f"out/{rank}.out"

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
        text = np.unique(
            np.fromiter(
                (str.lower(word) for word in word_filter_re.findall(text)),
                dtype=StringDType(),
            )
        )
        labels[label] += 1
        for word in text:
            words_to_counts[word][label] += 1


def get_sum(d):
    res = 0
    for v in d.values():
        res += v
    return res > 1


num_labels = np.array(max_label + 1, dtype=np.int32)
num_labels_req = comm.Iallreduce(MPI.IN_PLACE, num_labels, MPI.MAX)

counts = np.fromiter((get_sum(v) for v in words_to_counts.values()), dtype=np.int8)
words = np.fromiter(words_to_counts.keys(), dtype=StringDType())

# mpi --------------------------------------------------------------------------------

ind = words.argsort()
words = words[ind]
words_orig = words.copy()
counts = counts[ind]


words_recv_buf = bytearray(256)

str_cnt_out = np.zeros(2, dtype=np.int32)
str_cnt_in = np.zeros_like(str_cnt_out)
send_reqs = [MPI.REQUEST_NULL for _ in range(3)]
recv_reqs = [MPI.REQUEST_NULL for _ in range(2)]
i = 1
while i < world_size:
    recv_from = (rank - i) % world_size
    send_to = (rank + i) % world_size
    i *= 2
    words_send_buf = ",".join(words).encode()
    str_cnt_out[:] = [len(words_send_buf), counts.shape[0]]


    send_reqs[0] = comm.Isend(str_cnt_out, send_to)
    send_reqs[1] = comm.Isend(words_send_buf, send_to)
    send_reqs[2] = comm.Isend(counts, send_to)

    comm.Recv(str_cnt_in, recv_from)
    new_buf_len = len(words_recv_buf)
    while new_buf_len < str_cnt_in[0]:
        new_buf_len *= 2
    words_recv_buf[len(words_recv_buf) :] = b"\0" * (new_buf_len - len(words_recv_buf))
    recv_reqs[0] = comm.Irecv(words_recv_buf, recv_from)
    counts_recv_buf = np.zeros(str_cnt_in[1], dtype=np.int8)
    recv_reqs[1] = comm.Irecv(counts_recv_buf, recv_from)

    recv_reqs[0].Wait()
    words = np.concatenate([words, words_recv_buf[: str_cnt_in[0]].decode().split(",")])
    sort_ind = words.argsort(kind="stable")
    words = words[sort_ind]
    words, ind, counts_new = np.unique(words, return_index=True, return_counts=True)
    counts_new = counts_new > 1
    recv_reqs[1].Wait()
    counts = np.concatenate([counts, counts_recv_buf])
    counts = counts[sort_ind]
    counts = counts[ind]
    counts[counts_new] = 1
    MPI.Request.Waitall(send_reqs)


# final ---------------------------------------------------------
words = words[np.nonzero(counts)]

num_labels_req.Wait()

label_counts = np.zeros(num_labels, dtype=np.int32)
for label, count in labels.items():
    label_counts[label] = count

labels_req = comm.Iallreduce(MPI.IN_PLACE, label_counts)

res = np.zeros((num_labels, words.shape[0]), dtype=np.int32)
word_locs = np.isin(words, words_orig).nonzero()[0]

for loc, word in zip(word_locs, words[word_locs]):
    for label, count in words_to_counts[word].items():
        res[label][loc] = count


res_shape = res.shape[:]
res_type = res.dtype

res = csr_array(res)

data_in = np.zeros(res_shape, dtype=res_type).flatten()
indices_in = np.zeros_like(data_in, dtype=np.int32)
indptr_in = np.zeros_like(res.indptr, dtype=np.int32)

bufs_in = [data_in, indices_in, indptr_in]

i = 1
recv_stats = [MPI.Status() for _ in range(3)]
while i < world_size:
    send_reqs = []
    recv_reqs = []

    recv_from = (rank - i) % world_size
    send_to = (rank + i) % world_size
    send_reqs.append(comm.Isend(res.data, send_to))
    send_reqs.append(comm.Isend(res.indices, send_to))
    send_reqs.append(comm.Isend(res.indptr, send_to))

    recv_reqs.append(comm.Irecv(data_in, source=recv_from))
    recv_reqs.append(comm.Irecv(indices_in, source=recv_from))
    recv_reqs.append(comm.Irecv(indptr_in, source=recv_from))
    MPI.Request.Waitall(recv_reqs, recv_stats)
    temp_csr = csr_array(
        (
            data_in[: recv_stats[0].Get_count(MPI.INT32_T)],
            indices_in[: recv_stats[1].Get_count(MPI.INT32_T)],
            indptr_in[: recv_stats[2].Get_count(MPI.INT32_T)],
        ),
        shape=res_shape,
    )

    MPI.Request.Waitall(send_reqs)
    res += temp_csr
    i *= 2


res = res.astype(np.double).toarray()
# comm.Allreduce(MPI.IN_PLACE, res)
# res = res.astype(np.double)

labels_req.Wait()

label_counts = label_counts.astype(np.double)[:, np.newaxis]
label_zeros = label_counts == 0
label_counts[label_zeros] = 1.0

res /= label_counts

label_counts[label_zeros] = 0.0
label_sum = label_counts.sum()

label_counts /= label_sum
res = np.concatenate((res, label_counts), axis=1)

with open(fn_out, "w") as f:
    f.write(",".join(words))
    f.write("\n")
    np.savetxt(f, res, delimiter=",")
