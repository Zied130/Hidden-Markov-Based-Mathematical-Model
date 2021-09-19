import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score


split_sequences=True
word2idx = {}
tag2idx = {}
pos2idx = {}
word_idx = 0
tag_idx = 0
pos_idx = 0
Xtrain = []
Ytrain = []
Ptrain=[]
currentX = []
currentY = []
currentP=[]
for line in open('train_all.txt',encoding='utf-8'):
    line = line.rstrip()
    if line:
        r = line.split()
        word, tag, pos = r
        if word not in word2idx:
            word2idx[word] = word_idx
            word_idx += 1
        currentX.append(word2idx[word])

        if tag not in tag2idx:
            tag2idx[tag] = tag_idx
            tag_idx += 1
        currentY.append(tag2idx[tag])
        if pos not in pos2idx:
            pos2idx[pos] = pos_idx
            pos_idx += 1
        currentP.append(pos2idx[pos])
    elif split_sequences:
        Xtrain.append(currentX)
        Ytrain.append(currentY)
        Ptrain.append(currentP)
        currentX = []
        currentY = []
        currentP=[]

if not split_sequences:
    Xtrain = currentX
    Ytrain = currentY
    Ptrain=currentP

word2idx=tag2idx
Xtrain=Ytrain
Ytrain=Ptrain


V = len(word2idx) 

# find hidden state transition matrix and pi
M = max(max(y) for y in Ytrain) + 1 #len(set(flatten(Ytrain)))
A = np.ones((M, M)) 
pi = np.ones(M)
for y in Ytrain:
    pi[y[0]] += 1
    for i in range(len(y)-1):
        A[y[i], y[i+1]] += 1
# turn it into a probability matrix
A /= A.sum(axis=1, keepdims=True)
pi /= pi.sum()

# find the observation matrix
B = np.ones((M, V)) # add-one smoothing
for x, y in zip(Xtrain, Ytrain):
    for xi, yi in zip(x, y):
        B[yi, xi] += 1
B /= B.sum(axis=1, keepdims=True)

class HMM:
    def __init__(self, M,A,B,pi):
        self.M = M # number of hidden states
        self.A=A
        self.B=B
        
        self.pi=pi
        
        self.word2idx=word2idx
    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(self.B[:,x[0]])
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states


hmm = HMM(M,A,B,pi)

Ptrain = []
for x in Xtrain:
    p = hmm.get_state_sequence(x)
    Ptrain.append(p)


def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total

def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()

print("test accuracy:", accuracy(Ytrain, Ptrain))
print("test f1:", total_f1_score(Ytrain, Ptrain))







