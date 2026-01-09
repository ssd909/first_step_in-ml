from collections import Counter
import math

def n_ngrams(text, n):
    words = text.lower().split()
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def modified_precision(candidate,reference,n):
    ng_cand=n_ngrams(candidate,n)
    ng_ref=n_ngrams(reference,n)
    ng_cand_cn=Counter(ng_cand)
    ng_ref_cn=Counter(ng_ref)
    min_clip = 0
    for word, count in ng_cand_cn.items():
        min_clip+=min(count,ng_ref_cn.get(word,0))
    total_clip=max(1,len(ng_cand))
    return min_clip/total_clip

def brevity_penalty(candidate,reference):
    can_length=len(candidate.split())
    ref_length=len(reference.split())
    if can_length>ref_length:
        return  1
    else:
        return math.exp(1 - can_length / ref_length)

def blue_score(candidate,reference,n):
    ls_ps=list()
    for i in range(1,n+1):
        p=modified_precision(candidate,reference,i)
        ls_ps.append(p)
    geo_min=0
    if 0 in ls_ps:return geo_min


    else:
        score_sum = sum((1 / n) * math.log(p) for p in ls_ps)
        geo_mean = math.exp(score_sum)
        return geo_mean*brevity_penalty(candidate,reference)
ref = "კატა ზის დიდ მწვანე ბალახზე"
cand = "კატა ზის მწვანე ბალახზე"
print(blue_score(cand,ref,1))