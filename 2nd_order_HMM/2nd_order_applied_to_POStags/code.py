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

for L in Xtrain:
    L.insert(0,3)
for L in Ytrain:
    L.insert(0,3)
for L in Ptrain:
    L.insert(0,0)

V = len(tag2idx) 
M = len(pos2idx) 

pi=np.ones(V)

for y in Ytrain:
    pi[y[1]] += 1
pi /= pi.sum()

A=np.ones((M,M,M))

for p in Ptrain:
    i=0
    while i<len(p)-2:
        A[p[i],p[i+1],p[i+2]]+=1
        i=i+1
A /= A.sum(axis=(0,2), keepdims=True)
        
B=np.ones((M,M,V))

for y,p in zip(Ytrain,Ptrain):
    for i in range(len(y)-1):
        B[p[i],p[i+1],y[i+1]] += 1
B /= B.sum(axis=(0,2), keepdims=True)        

class HMM:
    def __init__(self, M,A,B,pi):
        
        self.M=M
        self.A=A
        self.B=B
        self.pi=pi
        
    def get_state_sequence(self, y):
        
        T = len(y)
        delta=np.zeros((T-1,M,M))
        psi=np.zeros((T-1,M,M))
        for i in range(self.M):
            for j in range(self.M):
                delta[0,i,j]=np.log(self.pi[i])+np.log(self.B[i,j,y[1]])
        for t in range(1,T-1):
            for k in range(self.M):
                for j in range(self.M):
                    delta[t,j,k]=np.max(delta[t-1,:,j]+np.log(self.A[:,j,k]))+np.log(self.B[j,k,y[t+1]])
                    psi[t,j,k]=np.argmax(delta[t-1,:,j]+np.log(self.A[:,j,k]))
            
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.unravel_index(np.argmax(delta[-1,:,:], axis=None), delta[-1,:,:].shape)[1]
        states[T-2] = np.unravel_index(np.argmax(delta[-1,:,:], axis=None), delta[-1,:,:].shape)[0]
        for t in range(T-3, -1, -1):
            states[t] = psi[t+1, states[t+1], states[t+2]]
        return states
        
hmm = HMM(M,A,B,pi)

P1train = []
for y in Ytrain:
    p = hmm.get_state_sequence(y)
    P1train.append(p)
    
    
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

print("test accuracy:", accuracy(Ptrain, P1train))
print("test f1:", total_f1_score(Ptrain, P1train))


        
        
        






















