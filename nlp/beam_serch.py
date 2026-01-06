import numpy as np
from math import log
import heapq
def beam_search(data,k):
    sequences = [[list(), 0.0]]

    for row in data:
        all_variant=list()
        for i in range(len(sequences)):
            idx,score=sequences[i]
            for j in range(len(row)):
                new_variant=[idx+[j],score-log(row[j])]
                all_variant.append(new_variant)
        sorted_variant=heapq.nlargest(k,all_variant,key=lambda x:x[1])
        sequences=sorted_variant
    return sequences

answer=beam_search([[0.1,0.23,0.22],[0.11,0.12,0.34]],3)
print(answer)