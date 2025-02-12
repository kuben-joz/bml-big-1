from mpi4py import MPI

comm = MPI.COMM_WORLD
if comm.rank == 0:
    comm.Send([b'hello',MPI.CHAR],dest=1)
elif comm.rank == 1:
    buf = bytearray(256)
    comm.Recv([buf,MPI.CHAR],source=0)
    # Decode the string to remove the empty characters from bytearray
    print(buf.decode('utf-8'))


a[:, a[0].argsort()]




import numpy as np
import re
from collections import defaultdict

word_filter_re = re.compile(r"[a-zA-Z]+")
line_split_re = re.compile(r"^(.*),(\d+)$", re.DOTALL)
# todo
# feature selection , mutual info https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
# 
words = defaultdict(lambda: 0)
with open('amazon_reviews_2M.csv', 'r') as f:
  for line in f:
    text, label = line_split_re.findall(line)[0]
    label = int(label)
    text = [str.encode(str.lower(word)) for word in word_filter_re.findall(text)]
    for word in text:
       words[word] += 1


num_single = 0
avg_length = 0.0
num_words = len(words)
max_word = 0
max_w = ""
min_word = 1000
for word, freq in words.items():
  w_len = len(word)
  avg_length += len(word) / w_len
  #max_word = max(max_word, w_len)
  if w_len > max_word:
    max_w = word
    max_word = w_len
  min_word = min(min_word, w_len)
  if freq == 1:
    num_single += 1

print(f"num words: {num_words}, single: {num_single}, multi:{num_words-num_single}")
print(f"avg len: {avg_length}, max: {max_word}, min:{min_word}")
print(max_w)