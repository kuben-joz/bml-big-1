import re
import sys
from collections import defaultdict

import numpy as np
from numpy.dtypes import StringDType

f_in = "in.csv"
f_out = "out.csv"
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


num_labels = np.array(max_label + 1, dtype=np.int32)

label_counts = np.zeros((num_labels, 1), dtype=np.double)
for label, count in labels.items():
    label_counts[label, 0] = count

zeros = np.zeros(num_labels, dtype=np.double)


def get_counts(d):
    zeros[:] = 0
    for label, count in d.items():
        zeros[label] = count
    return zeros


words = np.fromiter(
    (k for k, v in words_to_counts.items() if v[0] > 1),
    dtype=StringDType(),
)

res = np.fromiter(
    (get_counts(v[1]) for v in words_to_counts.values() if v[0] > 1),
    dtype=np.dtype((np.double, num_labels)),
    count=words.shape[0],
)


ind = words.argsort()
words = words[ind]
res = np.transpose(res[ind])

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
