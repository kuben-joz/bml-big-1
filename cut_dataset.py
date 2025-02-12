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
with open(fn, "r") as f_in:
    with open('amazon_cut.csv', 'w') as f_out:
      for i, line in enumerate(f_in):
          f_out.write(line)
          if i == 100_000:
             break
