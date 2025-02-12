import re
from collections import defaultdict

import numpy as np
from numpy.dtypes import StringDType

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)

fn = "amazon_reviews_2M.csv"
# fn = "test.csv"

words = defaultdict(lambda: 0)
max_word = 0
with open(fn, "r") as f:
    for i, line in enumerate(f):
        text, label = line_split_re.findall(line)[0]
        label = int(label)
        text = [str.encode(str.lower(word)) for word in word_filter_re.findall(text)]
        for word in text:
            max_word = max(max_word, len(word))
            words[word] += 1

arr = np.array([k for k in words.keys()], dtype=StringDType())
np.save("amazon.npy", arr=arr)