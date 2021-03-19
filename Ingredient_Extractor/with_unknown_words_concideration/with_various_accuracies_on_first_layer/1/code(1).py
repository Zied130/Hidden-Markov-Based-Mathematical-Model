import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score,accuracy_score
import math

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
for line in open('train1_all.txt',encoding='utf-8'):
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
    L.insert(0,Xtrain[0][-1])
for L in Ytrain:
    L.insert(0,Ytrain[0][-1])
for L in Ptrain:
    L.insert(0,Ptrain[0][-1])
    
    
V = len(word2idx) + 1
M = max(max(y) for y in Ytrain) + 1 

A = np.ones((M, M))
pi = np.ones(M)

for y in Ytrain:
    pi[y[0]] += 1
    for i in range(len(y)-1):
        A[y[i], y[i+1]] += 1
A /= A.sum(axis=1, keepdims=True)
pi /= pi.sum()

# find the observation matrix
B = np.ones((M, V)) # add-one smoothing
for x, y in zip(Xtrain, Ytrain):
    for xi, yi in zip(x, y):
        B[yi, xi] += 1
B /= B.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M,A,B,C,C1,pi,SUFF,SUFF1,word2idx):
        self.M = M # number of hidden states
        self.A=A
        self.B=B
        self.C=C
        self.C1=C1
        self.pi=pi
        self.SUFF=SUFF
        self.SUFF1=SUFF1
        self.word2idx=word2idx
    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        try:
            delta[0] = np.log(self.pi) + np.log(self.B[:,x[0]])
            
        except IndexError:
            try:
                delta[0] = np.log(self.pi) + np.log(self.C[:,SUFF.index([*word2idx][x[0]][:2])])
                
            except ValueError:
                delta[0] = np.log(self.pi) + np.log(self.C1[:,SUFF1.index([*word2idx][x[0]][:1])])
            except IndexError:
                try:
                    delta[0] = np.log(self.pi) + np.log(self.C1[:,SUFF1.index([*word2idx][x[0]][:1])])
                except IndexError:
                    delta[0] = np.log(self.pi)
                
        for t in range(1, T):
            for j in range(self.M):
                try:
                    delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, x[t]])
                    psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
                    
                except IndexError:
                    try:
                        delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.C[j, SUFF.index([*word2idx][x[t]][:2])])
                        psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
                        
                    except ValueError:
                        try:
                            delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.C1[j, SUFF1.index([*word2idx][x[t]][:1])])
                            psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
                        except ValueError:
                            delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j]))
                            psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
                    except IndexError:
                        try:
                            delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.C1[j, SUFF1.index([*word2idx][x[t]][:1])])
                            psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
                        except IndexError:
                            delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) 
                            psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))
        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states


SUFF=[]
SUFF1=[]
for w in [*word2idx]:
    SUFF.append(w[:2])
    SUFF1.append(w[:1])

suff_pos = defaultdict(list)
suff_pos1 = defaultdict(list)

idx=0
for suf in SUFF:
    suff_pos[suf].append(idx)
    idx+=1

idx=0
for suf in SUFF1:
    suff_pos1[suf].append(idx)
    idx+=1

C=np.ones((M,V))
C1=np.ones((M,V))


for l in suff_pos.values():
    C[:,l]=B[:,l].sum(axis=1, keepdims=True)/len(l)
    
for l in suff_pos1.values():
    C1[:,l]=B[:,l].sum(axis=1, keepdims=True)/len(l)

word_idx = len(word2idx)
w_known=len(word2idx)
word2idx_test={}
Xtest = []
currentX = []

for line in open('test1_all.txt',encoding='utf-8'):
    line = line.rstrip()
    if line:
        r = line.split()
        word = r[0]
        if word not in word2idx:
            word2idx_test[word] = word_idx
            word2idx[word]= word_idx
            word_idx += 1
        else:
            word2idx_test[word]=word2idx[word]
        currentX.append(word2idx_test[word])
    elif split_sequences:
            Xtest.append(currentX)
            currentX = []
            
            
for L in Xtest:
    L.insert(0,Xtest[0][-1])
    
    
hmm = HMM(M,A,B,C,C1,pi,SUFF,SUFF1,word2idx)

Y11test = []
for x in Xtest:
    y = hmm.get_state_sequence(x)
    Y11test.append(y)
    

Ytest=[]
list1=[]
for line in open('test1_all.txt',encoding='utf-8'):
    line = line.rstrip()
    if line:
        r = line.split()
        tag = r[1]
        list1.append(tag2idx[tag])
    elif split_sequences:
            Ytest.append(list1)
            list1 = []
            
for L in Ytest:
    L.insert(0,Ytest[0][-1])

def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total

def accuracy_unknown(T, Y,X):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y,x in zip(T, Y,X):
        for ti,yi,xi in zip (t,y,x):
            if xi>w_known :
                n_correct += (ti == yi)
                n_total += 1
    return float(n_correct) / (n_total+1)

def accuracy_known(T, Y,X):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y,x in zip(T, Y,X):
        for ti,yi,xi in zip (t,y,x):
            if xi<=w_known :
                n_correct += (ti == yi)
                n_total += 1
    return float(n_correct) / n_total

def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()

print("test accuracy:", accuracy(Y11test, Ytest))
accuracy=accuracy(Y11test, Ytest)
print("test f1:", total_f1_score(Y11test, Ytest))
f1=total_f1_score(Y11test, Ytest)
print("test accuracy for unknown words:",accuracy_unknown(Y11test, Ytest,Xtest))
unknown_ac=accuracy_unknown(Y11test, Ytest,Xtest)
print("test accuracy for known words:",accuracy_known(Y11test, Ytest,Xtest))
known_ac=accuracy_known(Y11test, Ytest,Xtest)

Y=np.concatenate(Ytest)
P=np.concatenate(Ytest)
Z=np.concatenate(Y11test)
X=np.concatenate(Xtest)


print("accuracy score for tag "+list(tag2idx.keys())[0]+" :", accuracy_score(Z[np.where(Y==0)[0]], P[np.where(Y==0)[0]]))
a11= accuracy_score(Z[np.where(Y==0)[0]], P[np.where(Y==0)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[1]+" :", accuracy_score(Z[np.where(Y==1)[0]], P[np.where(Y==1)[0]]))
a12= accuracy_score(Z[np.where(Y==1)[0]], P[np.where(Y==1)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[2]+" :", accuracy_score(Z[np.where(Y==2)[0]], P[np.where(Y==2)[0]]))
a13=accuracy_score(Z[np.where(Y==2)[0]], P[np.where(Y==2)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[3]+" :", accuracy_score(Z[np.where(Y==3)[0]], P[np.where(Y==3)[0]]))
a14=accuracy_score(Z[np.where(Y==3)[0]], P[np.where(Y==3)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[4]+" :",accuracy_score(Z[np.where(Y==4)[0]], P[np.where(Y==4)[0]]))
a15=accuracy_score(Z[np.where(Y==4)[0]], P[np.where(Y==4)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[5]+" :", accuracy_score(Z[np.where(Y==5)[0]], P[np.where(Y==5)[0]]))
a16=accuracy_score(Z[np.where(Y==5)[0]], P[np.where(Y==5)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[6]+" :", accuracy_score(Z[np.where(Y==6)[0]], P[np.where(Y==6)[0]]))
a17=accuracy_score(Z[np.where(Y==6)[0]], P[np.where(Y==6)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[7]+" :", accuracy_score(Z[np.where(Y==7)[0]], P[np.where(Y==7)[0]]))
a18=accuracy_score(Z[np.where(Y==7)[0]], P[np.where(Y==7)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[8]+" :", accuracy_score(Z[np.where(Y==8)[0]], P[np.where(Y==8)[0]]))
a19= accuracy_score(Z[np.where(Y==8)[0]], P[np.where(Y==8)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[9]+" :", accuracy_score(Z[np.where(Y==9)[0]], P[np.where(Y==9)[0]]))
a110= accuracy_score(Z[np.where(Y==9)[0]], P[np.where(Y==9)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[10]+" :", accuracy_score(Z[np.where(Y==10)[0]], P[np.where(Y==10)[0]]))
a111= accuracy_score(Z[np.where(Y==10)[0]], P[np.where(Y==10)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[11]+" :", accuracy_score(Z[np.where(Y==11)[0]], P[np.where(Y==11)[0]]))
a112= accuracy_score(Z[np.where(Y==11)[0]], P[np.where(Y==11)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[12]+" :", accuracy_score(Z[np.where(Y==12)[0]], P[np.where(Y==12)[0]]))
a113= accuracy_score(Z[np.where(Y==12)[0]], P[np.where(Y==12)[0]])
print("accuracy score for tag "+list(tag2idx.keys())[13]+" :", accuracy_score(Z[np.where(Y==13)[0]], P[np.where(Y==13)[0]]))          
a114= accuracy_score(Z[np.where(Y==13)[0]], P[np.where(Y==13)[0]]) 


print("test f1 for tag "+list(tag2idx.keys())[0]+" :", f1_score(Z[np.where(Y==0)[0]], P[np.where(Y==0)[0]], average=None).mean())
a21= f1_score(Z[np.where(Y==0)[0]], P[np.where(Y==0)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[1]+" :", f1_score(Z[np.where(Y==1)[0]], P[np.where(Y==1)[0]], average=None).mean())
a22= f1_score(Z[np.where(Y==1)[0]], P[np.where(Y==1)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[2]+" :", f1_score(Z[np.where(Y==2)[0]], P[np.where(Y==2)[0]], average=None).mean())
a23=f1_score(Z[np.where(Y==2)[0]], P[np.where(Y==2)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[3]+" :", f1_score(Z[np.where(Y==3)[0]], P[np.where(Y==3)[0]], average=None).mean())
a24=f1_score(Z[np.where(Y==3)[0]], P[np.where(Y==3)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[4]+" :", f1_score(Z[np.where(Y==4)[0]], P[np.where(Y==4)[0]], average=None).mean())
a25=f1_score(Z[np.where(Y==4)[0]], P[np.where(Y==4)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[5]+" :", f1_score(Z[np.where(Y==5)[0]], P[np.where(Y==5)[0]], average=None).mean())
a26=f1_score(Z[np.where(Y==5)[0]], P[np.where(Y==5)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[6]+" :", f1_score(Z[np.where(Y==6)[0]], P[np.where(Y==6)[0]], average=None).mean())
a27=f1_score(Z[np.where(Y==6)[0]], P[np.where(Y==6)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[7]+" :", f1_score(Z[np.where(Y==7)[0]], P[np.where(Y==7)[0]], average=None).mean())
a28=f1_score(Z[np.where(Y==7)[0]], P[np.where(Y==7)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[8]+" :", f1_score(Z[np.where(Y==8)[0]], P[np.where(Y==8)[0]], average=None).mean())
a29= f1_score(Z[np.where(Y==8)[0]], P[np.where(Y==8)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[9]+" :", f1_score(Z[np.where(Y==9)[0]], P[np.where(Y==9)[0]], average=None).mean())
a210= f1_score(Z[np.where(Y==9)[0]], P[np.where(Y==9)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[10]+" :", f1_score(Z[np.where(Y==10)[0]], P[np.where(Y==10)[0]], average=None).mean())
a211= f1_score(Z[np.where(Y==10)[0]], P[np.where(Y==10)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[11]+" :", f1_score(Z[np.where(Y==11)[0]], P[np.where(Y==11)[0]], average=None).mean())
a212= f1_score(Z[np.where(Y==11)[0]], P[np.where(Y==11)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[12]+" :", f1_score(Z[np.where(Y==12)[0]], P[np.where(Y==12)[0]], average=None).mean())
a213= f1_score(Z[np.where(Y==12)[0]], P[np.where(Y==12)[0]], average=None).mean()
print("test f1 for tag "+list(tag2idx.keys())[13]+" :", f1_score(Z[np.where(Y==13)[0]], P[np.where(Y==13)[0]], average=None).mean())          
a214= f1_score(Z[np.where(Y==13)[0]], P[np.where(Y==13)[0]], average=None).mean()            


print("accuracy for unknown words for tag "+list(tag2idx.keys())[0]+" :", accuracy_score(Z[np.where(Y==0)[0][X[np.where(Y==0)[0]]>w_known]],P[np.where(Y==0)[0][X[np.where(Y==0)[0]]>w_known]]))
a31= accuracy_score(Z[np.where(Y==0)[0][X[np.where(Y==0)[0]]>w_known]],P[np.where(Y==0)[0][X[np.where(Y==0)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[0]+" :",len(set(np.where(X[np.where(Y==0)[0]]>w_known)[0])))
a41= len(set(np.where(X[np.where(Y==0)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[1]+" :", accuracy_score(Z[np.where(Y==1)[0][X[np.where(Y==1)[0]]>w_known]],P[np.where(Y==1)[0][X[np.where(Y==1)[0]]>w_known]]))
a32= accuracy_score(Z[np.where(Y==1)[0][X[np.where(Y==1)[0]]>w_known]],P[np.where(Y==1)[0][X[np.where(Y==1)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[1]+" :",len(set(np.where(X[np.where(Y==1)[0]]>w_known)[0])))
a42= len(set(np.where(X[np.where(Y==1)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[2]+" :", accuracy_score(Z[np.where(Y==2)[0][X[np.where(Y==2)[0]]>w_known]],P[np.where(Y==2)[0][X[np.where(Y==2)[0]]>w_known]]))
a33= accuracy_score(Z[np.where(Y==2)[0][X[np.where(Y==2)[0]]>w_known]],P[np.where(Y==2)[0][X[np.where(Y==2)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[2]+" :",len(set(np.where(X[np.where(Y==2)[0]]>w_known)[0])))
a43= len(set(np.where(X[np.where(Y==2)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[3]+" :", accuracy_score(Z[np.where(Y==3)[0][X[np.where(Y==3)[0]]>w_known]],P[np.where(Y==3)[0][X[np.where(Y==3)[0]]>w_known]]))
a34= accuracy_score(Z[np.where(Y==3)[0][X[np.where(Y==3)[0]]>w_known]],P[np.where(Y==3)[0][X[np.where(Y==3)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[3]+" :",len(set(np.where(X[np.where(Y==3)[0]]>w_known)[0])))
a44= len(set(np.where(X[np.where(Y==3)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[4]+" :", accuracy_score(Z[np.where(Y==4)[0][X[np.where(Y==4)[0]]>w_known]],P[np.where(Y==4)[0][X[np.where(Y==4)[0]]>w_known]]))
a35= accuracy_score(Z[np.where(Y==4)[0][X[np.where(Y==4)[0]]>w_known]],P[np.where(Y==4)[0][X[np.where(Y==4)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[4]+" :",len(set(np.where(X[np.where(Y==4)[0]]>w_known)[0])))
a45= len(set(np.where(X[np.where(Y==4)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[5]+" :", accuracy_score(Z[np.where(Y==5)[0][X[np.where(Y==5)[0]]>w_known]],P[np.where(Y==5)[0][X[np.where(Y==5)[0]]>w_known]]))
a36= accuracy_score(Z[np.where(Y==5)[0][X[np.where(Y==5)[0]]>w_known]],P[np.where(Y==5)[0][X[np.where(Y==5)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[5]+" :",len(set(np.where(X[np.where(Y==5)[0]]>w_known)[0])))
a46= len(set(np.where(X[np.where(Y==5)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[6]+" :", accuracy_score(Z[np.where(Y==6)[0][X[np.where(Y==6)[0]]>w_known]],P[np.where(Y==6)[0][X[np.where(Y==6)[0]]>w_known]]))
a37= accuracy_score(Z[np.where(Y==6)[0][X[np.where(Y==6)[0]]>w_known]],P[np.where(Y==6)[0][X[np.where(Y==6)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[6]+" :",len(set(np.where(X[np.where(Y==6)[0]]>w_known)[0])))
a47= len(set(np.where(X[np.where(Y==6)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[7]+" :", accuracy_score(Z[np.where(Y==7)[0][X[np.where(Y==7)[0]]>w_known]],P[np.where(Y==7)[0][X[np.where(Y==7)[0]]>w_known]]))
a38=  accuracy_score(Z[np.where(Y==7)[0][X[np.where(Y==7)[0]]>w_known]],P[np.where(Y==7)[0][X[np.where(Y==7)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[7]+" :",len(set(np.where(X[np.where(Y==7)[0]]>608)[0])))
a48= len(set(np.where(X[np.where(Y==7)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[8]+" :", accuracy_score(Z[np.where(Y==8)[0][X[np.where(Y==8)[0]]>w_known]],P[np.where(Y==8)[0][X[np.where(Y==8)[0]]>w_known]]))
a39= accuracy_score(Z[np.where(Y==8)[0][X[np.where(Y==8)[0]]>w_known]],P[np.where(Y==8)[0][X[np.where(Y==8)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[8]+" :",len(set(np.where(X[np.where(Y==8)[0]]>w_known)[0])))
a49= len(set(np.where(X[np.where(Y==8)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[9]+" :", accuracy_score(Z[np.where(Y==9)[0][X[np.where(Y==9)[0]]>w_known]],P[np.where(Y==9)[0][X[np.where(Y==9)[0]]>w_known]]))
a310= accuracy_score(Z[np.where(Y==9)[0][X[np.where(Y==9)[0]]>w_known]],P[np.where(Y==9)[0][X[np.where(Y==9)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[9]+" :",len(set(np.where(X[np.where(Y==9)[0]]>w_known)[0])))
a410= len(set(np.where(X[np.where(Y==9)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[10]+" :", accuracy_score(Z[np.where(Y==10)[0][X[np.where(Y==10)[0]]>w_known]],P[np.where(Y==10)[0][X[np.where(Y==10)[0]]>w_known]]))
a311=accuracy_score(Z[np.where(Y==10)[0][X[np.where(Y==10)[0]]>w_known]],P[np.where(Y==10)[0][X[np.where(Y==10)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[10]+" :",len(set(np.where(X[np.where(Y==10)[0]]>w_known)[0])))
a411= len(set(np.where(X[np.where(Y==10)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[11]+" :", accuracy_score(Z[np.where(Y==11)[0][X[np.where(Y==11)[0]]>w_known]],P[np.where(Y==11)[0][X[np.where(Y==11)[0]]>w_known]]))
a312= accuracy_score(Z[np.where(Y==11)[0][X[np.where(Y==11)[0]]>w_known]],P[np.where(Y==11)[0][X[np.where(Y==11)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[11]+" :",len(set(np.where(X[np.where(Y==11)[0]]>w_known)[0])))
a412= len(set(np.where(X[np.where(Y==11)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[12]+" :", accuracy_score(Z[np.where(Y==12)[0][X[np.where(Y==12)[0]]>w_known]],P[np.where(Y==12)[0][X[np.where(Y==12)[0]]>w_known]]))
a313= accuracy_score(Z[np.where(Y==12)[0][X[np.where(Y==12)[0]]>w_known]],P[np.where(Y==12)[0][X[np.where(Y==12)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[12]+" :",len(set(np.where(X[np.where(Y==12)[0]]>w_known)[0])))
a413= len(set(np.where(X[np.where(Y==12)[0]]>w_known)[0]))
print("accuracy for unknown words for tag "+list(tag2idx.keys())[13]+" :", accuracy_score(Z[np.where(Y==13)[0][X[np.where(Y==13)[0]]>w_known]],P[np.where(Y==13)[0][X[np.where(Y==13)[0]]>w_known]]))          
a314= accuracy_score(Z[np.where(Y==13)[0][X[np.where(Y==13)[0]]>w_known]],P[np.where(Y==13)[0][X[np.where(Y==13)[0]]>w_known]])
print("number of unknown words for tag "+list(tag2idx.keys())[13]+" :",len(set(np.where(X[np.where(Y==13)[0]]>w_known)[0])))
a414= len(set(np.where(X[np.where(Y==13)[0]]>w_known)[0]))

print("accuracy for known words for tag "+list(tag2idx.keys())[0]+" :", accuracy_score(Z[np.where(Y==0)[0][X[np.where(Y==0)[0]]<=w_known]],P[np.where(Y==0)[0][X[np.where(Y==0)[0]]<=w_known]]))
a51= accuracy_score(Z[np.where(Y==0)[0][X[np.where(Y==0)[0]]<=w_known]],P[np.where(Y==0)[0][X[np.where(Y==0)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[0]+" :",len(set(np.where(X[np.where(Y==0)[0]]<=w_known)[0])))
a61= len(set(np.where(X[np.where(Y==0)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[1]+" :", accuracy_score(Z[np.where(Y==1)[0][X[np.where(Y==1)[0]]<=w_known]],P[np.where(Y==1)[0][X[np.where(Y==1)[0]]<=w_known]]))
a52= accuracy_score(Z[np.where(Y==1)[0][X[np.where(Y==1)[0]]<=w_known]],P[np.where(Y==1)[0][X[np.where(Y==1)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[1]+" :",len(set(np.where(X[np.where(Y==1)[0]]<=w_known)[0])))
a62= len(set(np.where(X[np.where(Y==1)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[2]+" :", accuracy_score(Z[np.where(Y==2)[0][X[np.where(Y==2)[0]]<=w_known]],P[np.where(Y==2)[0][X[np.where(Y==2)[0]]<=w_known]]))
a53= accuracy_score(Z[np.where(Y==2)[0][X[np.where(Y==2)[0]]<=w_known]],P[np.where(Y==2)[0][X[np.where(Y==2)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[2]+" :",len(set(np.where(X[np.where(Y==2)[0]]<=w_known)[0])))
a63= len(set(np.where(X[np.where(Y==2)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[3]+" :", accuracy_score(Z[np.where(Y==3)[0][X[np.where(Y==3)[0]]<=w_known]],P[np.where(Y==3)[0][X[np.where(Y==3)[0]]<=w_known]]))
a54= accuracy_score(Z[np.where(Y==3)[0][X[np.where(Y==3)[0]]<=w_known]],P[np.where(Y==3)[0][X[np.where(Y==3)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[3]+" :",len(set(np.where(X[np.where(Y==3)[0]]<=w_known)[0])))
a64=len(set(np.where(X[np.where(Y==3)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[4]+" :", accuracy_score(Z[np.where(Y==4)[0][X[np.where(Y==4)[0]]<=w_known]],P[np.where(Y==4)[0][X[np.where(Y==4)[0]]<=w_known]]))
a55= accuracy_score(Z[np.where(Y==4)[0][X[np.where(Y==4)[0]]<=w_known]],P[np.where(Y==4)[0][X[np.where(Y==4)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[4]+" :",len(set(np.where(X[np.where(Y==4)[0]]<=w_known)[0])))
a65=len(set(np.where(X[np.where(Y==4)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[5]+" :", accuracy_score(Z[np.where(Y==5)[0][X[np.where(Y==5)[0]]<=w_known]],P[np.where(Y==5)[0][X[np.where(Y==5)[0]]<=w_known]]))
a56= accuracy_score(Z[np.where(Y==5)[0][X[np.where(Y==5)[0]]<=w_known]],P[np.where(Y==5)[0][X[np.where(Y==5)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[5]+" :",len(set(np.where(X[np.where(Y==5)[0]]<=w_known)[0])))
a66=len(set(np.where(X[np.where(Y==5)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[6]+" :", accuracy_score(Z[np.where(Y==6)[0][X[np.where(Y==6)[0]]<=w_known]],P[np.where(Y==6)[0][X[np.where(Y==6)[0]]<=w_known]]))
a57= accuracy_score(Z[np.where(Y==6)[0][X[np.where(Y==6)[0]]<=w_known]],P[np.where(Y==6)[0][X[np.where(Y==6)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[6]+" :",len(set(np.where(X[np.where(Y==6)[0]]<=w_known)[0])))
a67=len(set(np.where(X[np.where(Y==6)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[7]+" :", accuracy_score(Z[np.where(Y==7)[0][X[np.where(Y==7)[0]]<=w_known]],P[np.where(Y==7)[0][X[np.where(Y==7)[0]]<=w_known]]))
a58= accuracy_score(Z[np.where(Y==7)[0][X[np.where(Y==7)[0]]<=w_known]],P[np.where(Y==7)[0][X[np.where(Y==7)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[7]+" :",len(set(np.where(X[np.where(Y==7)[0]]<=w_known)[0])))
a68=len(set(np.where(X[np.where(Y==7)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[8]+" :", accuracy_score(Z[np.where(Y==8)[0][X[np.where(Y==8)[0]]<=w_known]],P[np.where(Y==8)[0][X[np.where(Y==8)[0]]<=w_known]]))
a59= accuracy_score(Z[np.where(Y==8)[0][X[np.where(Y==8)[0]]<=w_known]],P[np.where(Y==8)[0][X[np.where(Y==8)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[8]+" :",len(set(np.where(X[np.where(Y==8)[0]]<=w_known)[0])))
a69=len(set(np.where(X[np.where(Y==8)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[9]+" :", accuracy_score(Z[np.where(Y==9)[0][X[np.where(Y==9)[0]]<=w_known]],P[np.where(Y==9)[0][X[np.where(Y==9)[0]]<=w_known]]))
a510= accuracy_score(Z[np.where(Y==9)[0][X[np.where(Y==9)[0]]<=w_known]],P[np.where(Y==9)[0][X[np.where(Y==9)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[9]+" :",len(set(np.where(X[np.where(Y==9)[0]]<=w_known)[0])))
a610=len(set(np.where(X[np.where(Y==9)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[10]+" :", accuracy_score(Z[np.where(Y==10)[0][X[np.where(Y==10)[0]]<=w_known]],P[np.where(Y==10)[0][X[np.where(Y==10)[0]]<=w_known]]))
a511= accuracy_score(Z[np.where(Y==10)[0][X[np.where(Y==10)[0]]<=w_known]],P[np.where(Y==10)[0][X[np.where(Y==10)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[10]+" :",len(set(np.where(X[np.where(Y==10)[0]]<=w_known)[0])))
a611=len(set(np.where(X[np.where(Y==10)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[11]+" :", accuracy_score(Z[np.where(Y==11)[0][X[np.where(Y==11)[0]]<=w_known]],P[np.where(Y==11)[0][X[np.where(Y==11)[0]]<=w_known]]))
a512= accuracy_score(Z[np.where(Y==11)[0][X[np.where(Y==11)[0]]<=w_known]],P[np.where(Y==11)[0][X[np.where(Y==11)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[11]+" :",len(set(np.where(X[np.where(Y==11)[0]]<=w_known)[0])))
a612=len(set(np.where(X[np.where(Y==11)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[12]+" :", accuracy_score(Z[np.where(Y==12)[0][X[np.where(Y==12)[0]]<=w_known]],P[np.where(Y==12)[0][X[np.where(Y==12)[0]]<=w_known]]))
a513= accuracy_score(Z[np.where(Y==12)[0][X[np.where(Y==12)[0]]<=w_known]],P[np.where(Y==12)[0][X[np.where(Y==12)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[12]+" :",len(set(np.where(X[np.where(Y==12)[0]]<=w_known)[0])))
a613=len(set(np.where(X[np.where(Y==12)[0]]<=w_known)[0]))
print("accuracy for known words for tag "+list(tag2idx.keys())[13]+" :", accuracy_score(Z[np.where(Y==13)[0][X[np.where(Y==13)[0]]<=w_known]],P[np.where(Y==13)[0][X[np.where(Y==13)[0]]<=w_known]]))          
a514= accuracy_score(Z[np.where(Y==13)[0][X[np.where(Y==13)[0]]<=w_known]],P[np.where(Y==13)[0][X[np.where(Y==13)[0]]<=w_known]])
print("number of known words for tag "+list(tag2idx.keys())[13]+" :",len(set(np.where(X[np.where(Y==13)[0]]<=w_known)[0])))
a614=len(set(np.where(X[np.where(Y==13)[0]]<=w_known)[0]))

A=[[a11,a21,a31,a41,a51,a61],[a12,a22,a32,a42,a52,a62],[a13,a23,a33,a43,a53,a63],[a14,a24,a34,a44,a54,a64]
,[a15,a25,a35,a45,a55,a65],[a16,a26,a36,a46,a56,a66],[a17,a27,a37,a47,a57,a67],[a18,a28,a38,a48,a58,a68],
[a19,a29,a39,a49,a59,a69],[a110,a210,a310,a410,a510,a610],[a111,a211,a311,a411,a511,a611],[a112,a212,a312,a412,a512,a612],
[a113,a213,a313,a413,a513,a613],[a114,a214,a314,a414,a514,a614]]


table_11 = pd.DataFrame(A,
columns=['accuracy', 'f1_score', 'accuracy for unknown words',
         'number of unknown words','accuracy for known words','number of known words']
,index=[list(tag2idx.keys())[0], list(tag2idx.keys())[1], list(tag2idx.keys())[2] , list(tag2idx.keys())[3] 
, list(tag2idx.keys())[4] , list(tag2idx.keys())[5],list(tag2idx.keys())[6],list(tag2idx.keys())[7]
,list(tag2idx.keys())[8],list(tag2idx.keys())[9],list(tag2idx.keys())[10],list(tag2idx.keys())[11],
list(tag2idx.keys())[12],list(tag2idx.keys())[13]])

str_pythontex=[float("{0:.2f}".format(list(table_11.loc["A"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["A"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["A"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["A"])[4]*100)),
round(list(table_11.loc["A"])[3]),round(list(table_11.loc["A"])[5]),
float("{0:.2f}".format(list(table_11.loc["B"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["B"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["B"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["B"])[4]*100)),
round(list(table_11.loc["B"])[3]),round(list(table_11.loc["B"])[5]),
float("{0:.2f}".format(list(table_11.loc["C"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["C"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["C"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["C"])[4]*100)),
round(list(table_11.loc["C"])[3]),round(list(table_11.loc["C"])[5]),
float("{0:.2f}".format(list(table_11.loc["D"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["D"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["D"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["D"])[4]*100)),
round(list(table_11.loc["D"])[3]),round(list(table_11.loc["D"])[5]),
float("{0:.2f}".format(list(table_11.loc["E"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["E"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["E"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["E"])[4]*100)),
round(list(table_11.loc["E"])[3]),round(list(table_11.loc["E"])[5]),
float("{0:.2f}".format(list(table_11.loc["F"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["F"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["F"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["F"])[4]*100)),
round(list(table_11.loc["F"])[3]),round(list(table_11.loc["F"])[5]),
float("{0:.2f}".format(list(table_11.loc["G"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["G"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["G"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["G"])[4]*100)),
round(list(table_11.loc["G"])[3]),round(list(table_11.loc["G"])[5]),
float("{0:.2f}".format(list(table_11.loc["H"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["H"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["H"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["H"])[4]*100)),
round(list(table_11.loc["H"])[3]),round(list(table_11.loc["H"])[5]),
float("{0:.2f}".format(list(table_11.loc["I"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["I"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["I"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["I"])[4]*100)),
round(list(table_11.loc["I"])[3]),round(list(table_11.loc["I"])[5]),
float("{0:.2f}".format(list(table_11.loc["J"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["J"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["J"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["J"])[4]*100)),
round(list(table_11.loc["J"])[3]),round(list(table_11.loc["J"])[5]),
float("{0:.2f}".format(list(table_11.loc["K"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["K"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["K"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["K"])[4]*100)),
round(list(table_11.loc["K"])[3]),round(list(table_11.loc["K"])[5]),
float("{0:.2f}".format(list(table_11.loc["L"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["L"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["L"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["L"])[4]*100)),
round(list(table_11.loc["L"])[3]),round(list(table_11.loc["L"])[5]),
float("{0:.2f}".format(list(table_11.loc["M"])[0]*100)),float("{0:.2f}".format(list(table_11.loc["M"])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["M"])[2]*100)),float("{0:.2f}".format(list(table_11.loc["M"])[4]*100)),
round(list(table_11.loc["M"])[3]),round(list(table_11.loc["M"])[5]),
float("{0:.2f}".format(list(table_11.loc["."])[0]*100)),float("{0:.2f}".format(list(table_11.loc["."])[1]*100)),
float("{0:.2f}".format(list(table_11.loc["."])[2]*100)),float("{0:.2f}".format(list(table_11.loc["."])[4]*100)),
round(list(table_11.loc["."])[3]),round(list(table_11.loc["."])[5]),
float("{0:.2f}".format(float(accuracy)*100)),float("{0:.2f}".format(float(f1)*100)),
float("{0:.2f}".format(float(unknown_ac)*100)),float("{0:.2f}".format(float(known_ac)*100)),
len(word2idx)-w_known,float("{0:.2f}".format(float((len(word2idx)-w_known)/len(word2idx))*100))]

L=[]
for x in str_pythontex:
    if math.isnan(x):
        L.append('NULL')
    else:
        L.append(str(x))

L1=[]
i=0
for x in L:
    i=i+1
    if i!=5 and i!=6 and x!="NULL":
        L1.append(x+" \%")
    elif x=="NULL":
        L1.append(x)
    elif i==5:
        L1.append(x)
    else:
        L1.append(x)
        i=0

v1=[float("{0:.2f}".format(float(accuracy)*100)),float("{0:.2f}".format(float(f1)*100)),
float("{0:.2f}".format(float(unknown_ac)*100)),float("{0:.2f}".format(float(known_ac)*100)),
len(word2idx)-w_known,float("{0:.2f}".format(float((len(word2idx)-w_known)/len(word2idx))*100))]


