import re
import sys
from collections import defaultdict

import numpy as np
from mpi4py import MPI
from numpy.dtypes import StringDType
from scipy.sparse import csr_array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


f_in = f"data/{rank}.csv"
f_out = f"out/1-{rank}.out"
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
num_labels_req = comm.Iallreduce(MPI.IN_PLACE, num_labels, MPI.MAX)


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


words_recv_buf = bytearray(256)

str_cnt_out = np.zeros(2, dtype=np.int32)
str_cnt_in = np.zeros_like(str_cnt_out)
send_reqs = [MPI.REQUEST_NULL for _ in range(3)]
recv_reqs = [MPI.REQUEST_NULL for _ in range(2)]
i = 1
recv_from = (rank - i) % world_size
send_to = (rank + i) % world_size

words_send_buf = ",".join(words).encode()
str_cnt_out[:] = [len(words_send_buf), counts.shape[0]]
send_reqs[0] = comm.Isend(str_cnt_out, send_to)
send_reqs[1] = comm.Isend(words_send_buf, send_to)
send_reqs[2] = comm.Isend(counts, send_to)


while i < world_size:
    i *= 2
    send_to = (rank + i) % world_size

    comm.Recv(str_cnt_in, recv_from)
    new_buf_len = len(words_recv_buf)
    while new_buf_len < str_cnt_in[0]:
        new_buf_len *= 2
    words_recv_buf[len(words_recv_buf) :] = b"\0" * (new_buf_len - len(words_recv_buf))
    recv_reqs[0] = comm.Irecv(words_recv_buf, recv_from)
    counts_recv_buf = np.zeros((str_cnt_in[1], bit_len), dtype=bit_type)
    recv_reqs[1] = comm.Irecv(counts_recv_buf, recv_from)

    recv_reqs[0].Wait()
    words = np.concatenate([words, words_recv_buf[: str_cnt_in[0]].decode().split(",")])
    # presorting with timsort is faster above ~250k line input per mpi proc
    sort_ind = words.argsort(kind="stable")
    words = words[sort_ind]
    words, counts_new = np.unique(words, return_counts=True)
    if i < world_size:
        words_send_buf = ",".join(words).encode()
        send_reqs[0].Wait()
        str_cnt_out[:] = [len(words_send_buf), counts_new.shape[0]]
        send_reqs[0] = comm.Isend(str_cnt_out, send_to)
        send_reqs[1].Wait()
        send_reqs[1] = comm.Isend(words_send_buf, send_to)
    counts_new = np.add.accumulate(counts_new)
    # counts_new = np.roll(counts_new, 1)
    counts_new[1:] = counts_new[:-1]
    counts_new[0] = 0
    recv_reqs[1].Wait()
    counts = np.concatenate([counts, counts_recv_buf])
    counts = counts[sort_ind]
    counts = np.bitwise_or.reduceat(counts, counts_new)
    if i < world_size:
        send_reqs[2].Wait()
        send_reqs[2] = comm.Isend(counts, send_to)
    recv_from = (rank - i) % world_size

MPI.Request.Waitall(send_reqs)

if rank == 0:
    print("finished exchange of strings")

# final ---------------------------------------------------------
counts = np.bitwise_count(counts)
# this should upcast automatically
counts = np.add.reduce(counts, axis=1) > 1
words = words[counts]

num_labels_req.Wait()

label_counts = np.zeros(num_labels, dtype=np.int32)
for label, count in labels.items():
    label_counts[label] = count

labels_req = comm.Iallreduce(MPI.IN_PLACE, label_counts)

res = np.zeros((words.shape[0], num_labels), dtype=np.int32)
word_locs = np.isin(words, words_orig).nonzero()[0]

for loc, word in zip(word_locs, words[word_locs]):
    for label, count in words_to_counts[word][1].items():
        res[loc][label] = count

res = np.transpose(res)

res_shape = res.shape[:]
res_size = res.size

# data+indicies+indptr+3 minus ones for splitting
buf_size = res_size + res_size + res_shape[0] + 1 + 3

buf = np.zeros(buf_size, dtype=np.int32)

res = csr_array(res)


def to_csr(buf):
    inters = (buf == -1).nonzero()[0]
    res = csr_array(
        (
            buf[: inters[0]],
            buf[inters[0] + 1 : inters[1]],
            buf[inters[1] + 1 : inters[2]],
        ),
        shape=res_shape,
    )
    return res


def to_buf(csr_arr, buf):
    i = csr_arr.data.size
    buf[:i] = csr_arr.data
    buf[i] = -1
    i_prev = i + 1
    i = csr_arr.indices.size + i_prev
    buf[i_prev:i] = csr_arr.indices
    buf[i] = -1
    i_prev = i + 1
    i = csr_arr.indptr.size + i_prev
    buf[i_prev:i] = csr_arr.indptr
    buf[i] = -1
    i += 1
    # todo check this is unnecessary
    buf[i:] = 0
    return i


# tree reduce then braodcast as allreduce, cloud do better but it's a pain in python
i = 1
while i < world_size:
    if rank % i != 0:
        break
    if rank % (i * 2) == 0:
        # recieves
        if rank + i >= world_size:
            # rank doesn't exist so we wait for the send in future iterations
            i *= 2
            continue
        recv_from = (rank + i) % world_size
        buf[:] = 0
        comm.Recv(buf, recv_from)
        res += to_csr(buf)
    else:
        # sends
        send_to = (rank - i) % world_size
        send_len = to_buf(res, buf)
        comm.Send(buf[:send_len], send_to)
    i *= 2

send_len = np.array([0], dtype=np.int32)
if rank == 0:
    send_len[0] = to_buf(res, buf)
    #comm.Bcast(send_len, root=0)
    #comm.Bcast(buf[:send_len[0]], root=0)
    comm.Bcast(buf, root=0)
    print("finished exchange data")
else:
    buf[:] = 0
    #comm.Bcast(send_len, root=0)
    #comm.Bcast(buf[:send_len[0]], root=0)
    comm.Bcast(buf, root=0)
    res = to_csr(buf)


res = res.astype(np.double).toarray()
# print(res_shape)
# comm.Barrier()
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

with open(f_out, "w") as f:
    f.write(",".join(words))
    f.write("\n")
    np.savetxt(f, res, delimiter=",")
