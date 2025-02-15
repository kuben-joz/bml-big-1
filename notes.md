num words: 266267, single: 128891

num words: 266267, single: 128891, multi:137376
avg len: 266267.0, max: 110, min:1
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


- can't get argsort and sort at the same time so we do create a copy, sorting by cycle would help but it's pretty slow without numba
- oculd use patricia trees and send python pickles instead, but that would work best if you could write the patricia tree in c


https://dabeaz.blogspot.com/2010/01/few-useful-bytearray-tricks.html

b = bytearray(10)
b[len(b):] = b'\0' * 10


https://stackoverflow.com/questions/5347065/interleaving-two-numpy-arrays-efficiently



The vocabs are partially sorted so to reduce latency we just all_gather and use timsort to merge the blocks locally, nothing much to overlap though