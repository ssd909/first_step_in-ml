import numpy as np
import re
txt= "/home/amirani/Desktop/txt"
with open(txt) as f:
  text=f.read()

new_tx=text.lower()
text1 = re.sub(r'\[\d+\]', '', new_tx)
text2 = re.sub(r'[^a-z\s]', '', new_tx)
text3=text2.split()
sorted_tx=sorted(list(set(text3)))

word2idx={word:idx for word,idx in enumerate(sorted_tx)}
idx2word={idx:word for word,idx in enumerate(sorted_tx)}
embedding_matrix = np.random.uniform(low=-0.5,high=0.5,size=(len(word2idx)+1,10))
new_step="The application of ML to business problems is known as predictive analytics"
sz=new_step.lower().split()
print(embedding_matrix)
print("***")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(1,len(sz)-1):
    target=sz[i]
    back_word=sz[i-1]
    nex_word=sz[i+1]

    all_idx=np.arange(len(sorted_tx))

    all_idx1=all_idx[all_idx!=i]
    ng_idx=np.random.choice(all_idx,size=5,replace=False)

    pos_score1=np.dot(embedding_matrix[idx2word[target]],embedding_matrix[idx2word[back_word]])


    pos_score2 = np.dot(embedding_matrix[idx2word[target]], embedding_matrix[idx2word[nex_word]])

    log_loss1=np.log(sigmoid(pos_score1))
    log_loss2=np.log(sigmoid(pos_score2))
    poss_loss=log_loss1+log_loss2
    ng_matrix=embedding_matrix[ng_idx]
    ng_loss=np.sum(np.log(sigmoid(-np.dot(ng_matrix,embedding_matrix[idx2word[target]]))))
    total_loss=-(poss_loss+ng_loss)
    old_target=embedding_matrix[idx2word[target]].copy()
    old_nex_word=embedding_matrix[idx2word[nex_word]].copy()
    old_back_word=embedding_matrix[idx2word[back_word]].copy()
    embedding_matrix[idx2word[nex_word]]-=0.001*(sigmoid(pos_score2)-1)*old_target
    embedding_matrix[idx2word[back_word]]-=0.001*(sigmoid(pos_score1)-1)*old_target
    embedding_matrix[idx2word[target]]-=0.001*(sigmoid(pos_score1)-1)*old_back_word
    embedding_matrix[idx2word[target]]-=0.001*(sigmoid(pos_score2)-1)*old_nex_word
    for k in ng_idx:

        ng_score=sigmoid(np.dot(embedding_matrix[k],old_target))

        embedding_matrix[idx2word[target]]-=0.001*ng_score*embedding_matrix[k]
        embedding_matrix[k] -= 0.001 * ng_score * old_target
print(embedding_matrix)