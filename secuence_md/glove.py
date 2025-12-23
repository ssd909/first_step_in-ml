import numpy as np

import re
url_txt = "/home/amirani/Desktop/txt"
with open(url_txt) as f:
  text=f.read()

new_tx=text.lower()
text1 = re.sub(r'\[\d+]', '', new_tx)
text2 = re.sub(r'[^a-z\s]', '', new_tx)
text3=text2.split()
def f_weight(x,x_max=100,alpa=0.75):
    if x<x_max:return (x/x_max)**alpa
    return 1
sorted_tx=sorted(list(set(text3)))

ctc_matrix=np.zeros((len(sorted_tx),len(sorted_tx)))
word2idx={word:i for i,word in enumerate(sorted_tx)}
for i,word in enumerate(text3):
    target_idx=word2idx[word]
    st=max(0,i-2)
    end=min(i+3,len(text3))
    for j in range(st,end):
        if i==j:continue
        cont_idx=word2idx[text3[j]]
        ctc_matrix[target_idx,cont_idx]+=1
embedding_matrix = np.random.uniform(low=-0.5,high=0.5,size=(len(word2idx)+1,10))
baias=np.zeros(len(word2idx))
lr=0.001
for k in range(1000):
    total_loss=0
    indexs=np.argwhere(ctc_matrix>0)
    for i,j in indexs:
        x_i_j=ctc_matrix[i,j]
        x_i=embedding_matrix[i]
        x_j=embedding_matrix[j]
        pred=np.dot(x_i,x_j)+baias[i]+baias[j]
        log_xij=np.log(x_i_j)
        err=pred-log_xij
        wg=f_weight(x_i_j)
        loss=wg*(err**2)
        total_loss+=loss
        embedding_matrix[i]-=lr*wg*x_j*err
        embedding_matrix[j]-=lr*wg*x_i*err
        baias[i]-=lr*wg*err
        baias[j]-=lr*wg*err
print(embedding_matrix)