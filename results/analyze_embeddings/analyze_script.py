import pickle
import gensim
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


BINS = 10
WORD_COUNT = 1000

(X,ids,reliability) = pickle.load(open(r'D:\JanneK\Documents\git_repos\LaureaTextAnalyzer\results\analyze_embeddings\word2vec_document_vectors.pickle','rb'))

ind = np.argsort(reliability)
reliability = reliability[ind]
ids = np.array(ids)[ind]
X = X[ind,:]
prc_score = np.array([stats.percentileofscore(reliability, a, 'rank') for a in reliability])

X_pooled = np.zeros((BINS,300))
reliability_pooled = np.zeros(BINS)
doc_count = np.zeros(BINS)

k=0
s = 100/BINS
for i in range(0,BINS):
    target=s*(i+1)
    k1=k
    while k<len(reliability)-1:
        k+=1
        if prc_score[k]>=target:
            break
    if i==0:
        X_pooled[i,:] = np.mean(X[k1:(k+1),:],axis=0)
        reliability_pooled[i] = np.mean(reliability[k1:(k + 1)])
        print('  %.2f (%i,%i);  ' % (reliability_pooled[i], k1, k),end='')
        count = k+1
    elif i==BINS-2:
        k-=1
        X_pooled[i, :] = np.mean(X[(k1+1):(k + 1), :], axis=0)
        reliability_pooled[i] = np.mean(reliability[(k1 + 1):(k + 1)])
        print('  %.2f (%i,%i);  ' % (reliability_pooled[i], k1+1, k),end='')
        count=k-k1
    else:
        X_pooled[i, :] = np.mean(X[(k1+1):(k + 1), :], axis=0)
        reliability_pooled[i] = np.mean(reliability[(k1 + 1):(k + 1)])
        print('  %.2f (%i,%i);  ' % (reliability_pooled[i], k1+1, k),end='')
        count=k-k1
    doc_count[i] = count
    print(' count=%i' % count)
print('\n')

filename = r'D:\JanneK\Documents\git_repos\LaureaTextAnalyzer\data\external\fin-word2vec.bin'

print('..reading embedding model')
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)  # you can continue training with the loaded model!
print('..done')

model.init_sims(replace=True)
print('Model has %i terms' % len(model.wv.vocab))

words=[]
word_sets=[]
for i in range(0,BINS):
    words.append(model.similar_by_vector(X_pooled[i,:],WORD_COUNT))
    print('bin %i with %i texts, mean score %f:\n   ' % (i+1,doc_count[i],reliability_pooled[i],),end='')
    for k in range(len(words[-1])):
        print('%s, ' % (words[-1][k][0],),end='')
    print('\n   [',end='')
    for k in range(len(words[-1])):
        print('%.3f, ' % (words[-1][k][1],),end='')
    print(']')
    word_sets.append([x[0] for x in words[-1]])

overlap = np.zeros(WORD_COUNT)
mat = np.zeros((BINS,BINS,WORD_COUNT))
mat_vec = np.zeros((BINS,BINS))
for i in range(0,BINS):
    for j in range(0, BINS):
        for z in range(WORD_COUNT):
            mat[i,j,z] = len(set(word_sets[i][0:(z+1)]).intersection(set(word_sets[j][0:(z+1)])))
            if i==0 and j==BINS-1:
                overlap[z]=mat[i,j,z]
        mat_vec[i, j] = cosine(X_pooled[i,:],X_pooled[j,:])

print(overlap)
print(mat[:,:,19])
print(mat_vec)

plt.plot(range(1,WORD_COUNT+1),overlap)
plt.show()


# print('\n\n')
#
# words_norm=[]
# for i in range(0,BINS):
#     words_norm.append(model.similar_by_vector(X_pooled[i,:]/np.linalg.norm(X_pooled[i,:]),10))
#     print('POOLED bin %i with mean score %f:\n   ' % (i+1,reliability_pooled[i],),end='')
#     for k in range(len(words_norm[-1])):
#         print('%s: %.3f, ' % (words_norm[-1][k][0],words_norm[-1][k][1]),end='')
#     print('')