


import re
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have",
  "Ain't": "Am not",
  "Aren't": "Are not",
  "Can't": "Cannot",
  "Can't've": "Cannot have",
  "Could've": "Could have",
  "Couldn't": "Could not",
  "Couldn't've": "Could not have",
  "Didn't": "Did not",
  "Doesn't": "Does not",
  "Don't": "Do not",
  "Hadn't": "Had not",
  "Hadn't've": "Had not have",
  "Hasn't": "Has not",
  "Haven't": "Have not",
  "He'd": "He would",
  "He'd've": "He would have",
  "He'll": "He will",
  "He'll've": "He will have",
  "He's": "He is",
  "How'd": "How did",
  "How'd'y": "How do you",
  "How'll": "How will",
  "How's": "How is",
  "Isn't": "Is not",
  "It'd": "It had",
  "It'd've": "It would have",
  "It'll": "It will",
  "It'll've": "It will have",
  "It's": "It is",
  "Let's": "Let us",
  "Ma'am": "Madam",
  "Mayn't": "May not",
  "Might've": "Might have",
  "Mightn't": "Might not",
  "Mightn't've": "Might not have",
  "Must've": "Must have",
  "Mustn't": "Must not",
  "Mustn't've": "Must not have",
  "Needn't": "Need not",
  "Needn't've": "Need not have",
  "Oughtn't": "Ought not",
  "Oughtn't've": "Ought not have",
  "Shan't": "Shall not",
  "Sha'n't": "Shall not",
  "Shan't've": "Shall not have",
  "She'd": "She would",
  "She'd've": "She would have",
  "She'll": "She will",
  "She'll've": "She will have",
  "She's": "She is",
  "Should've": "Should have",
  "Shouldn't": "Should not",
  "Shouldn't've": "Should not have",
  "So've": "So have",
  "So's": "So is",
  "That'd": "That would",
  "That'd've": "That would have",
  "That's": "That is",
  "There'd": "There had",
  "There'd've": "There would have",
  "There's": "There is",
  "They'd": "They would",
  "They'd've": "They would have",
  "They'll": "They will",
  "They'll've": "They will have",
  "They're": "They are",
  "They've": "They have",
  "To've": "To have",
  "Wasn't": "Was not",
  "We'd": "We ad",
  "We'd've": "We would have",
  "We'll": "We will",
  "We'll've": "We will have",
  "We're": "We are",
  "We've": "We have",
  "Weren't": "Were not",
  "What'll": "What will",
  "What'll've": "What will have",
  "What're": "What are",
  "What's": "What is",
  "What've": "What have",
  "When's": "When is",
  "When've": "When have",
  "Where'd": "Where did",
  "Where's": "Where is",
  "Where've": "Where have",
  "Who'll": "Who will",
  "Who'll've": "Who will have",
  "Who's": "Who is",
  "Who've": "Who have",
  "Why's": "Why is",
  "Why've": "Why have",
  "Will've": "Will have",
  "Won't": "Will not",
  "Won't've": "Will not have",
  "Would've": "Would have",
  "Wouldn't": "Would not",
  "Wouldn't've": "Would not have",
  "Y'all": "You all",
  "Y'alls": "You alls",
  "Y'all'd": "You all would",
  "Y'all'd've": "You all would have",
  "Y'all're": "You all are",
  "Y'all've": "You all have",
  "You'd": "You had",
  "You'd've": "You would have",
  "You'll": "You you will",
  "You'll've": "You you will have",
  "You're": "You are",
  "You've": "You have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


import pickle
import jsonlines
heads = []
descs = []
keywords = []
with jsonlines.open('data/sample-1M' + '.jsonl','r') as line:
    i = 0
    for art in line:
        if i < 10000:
            i += 1
            head = expandContractions(art["title"].replace('’','\''))
            desc = expandContractions(art["content"].replace('’','\''))
            
            head = head.replace("“","\"")
            desc = desc.replace("“","\"")
            
            head = head.replace("”","\"")
            desc = desc.replace("”","\"")
            
            head = head.replace("‘","'")
            desc = desc.replace("‘","'")
            
            rx = r'[|\{}\[\]"\':;?/<>.,+=\-()*&%$!]\b'
            rx2 = r'\b[|\{}\[\]"\':;?/<>.,+=\-()*&%$!]'
            head=re.sub(rx, ' \g<0> ', head)
            head=re.sub(rx2, ' \g<0> ', head)
            head=re.sub(' +', ' ', head)   
            
            desc=re.sub(rx, ' \g<0> ', desc)
            desc=re.sub(rx2, ' \g<0> ', desc)
            desc=re.sub(' +', ' ', desc)
            
            
            head = head.replace("' s ","'s ")
            desc = desc.replace("' s ","'s ")
            
            
            heads.append(head)
            descs.append(desc)
            keywords.append(None)
        else:
            break
        
with open('data/tokens.pkl', 'wb') as f:
    pickle.dump((heads,descs,keywords),f)

from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))
    return vocab, vocabcount

vocab, vocabcount = get_vocab(heads+descs)

len(vocab)

empty = 0
eos = 1
s_idx = eos+1

def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+s_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    
    idx2word = dict((idx,word) for word,idx in word2idx.items())

    return word2idx, idx2word

word2idx, idx2word = get_idx(vocab, vocabcount)

dim = 100

from keras.utils.data_utils import get_file
fname = 'glove.6B.%dd.txt'%dim
import os
datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
datadir = os.path.join(datadir_base, 'datasets')
gnm = os.path.join(datadir, fname)


gns = 400000
gns = int(gns[0].split()[0])
gns

import numpy as np
gid = {}
gew = np.empty((gns, dim))


globale_scale=.1
with open(gnm, encoding="utf8") as fp:
    i = 0
    for l in fp:
        l = l.strip().split()
        w = l[0]
        gid[w] = i
        gew[i,:] = list(map(float,l[1:]))
        i += 1
gew *= globale_scale

for w,i in gid.items():
    w = w.lower()
    if w not in gid:
        gid[w] = i

seed=42
vsize=40000
np.random.seed(seed)
shape = (vsize, dim)
scale = gew.std()*np.sqrt(12)/2 
embedding = np.random.uniform(low=-scale, high=scale, size=shape)

c = 0
for i in range(vsize):
    w = idx2word[i]
    g = gid.get(w, gid.get(w.lower()))
    if g is not None:
        embedding[i,:] = gew[g,:]
        c+=1





#print(c)
thro = 0.5

w2g = {}
for w in word2idx:
    if w in gid:
        g = w
    elif w.lower() in gid:
        g = w.lower()
    else:
        continue
    w2g[w] = g

normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []
for w,idx in word2idx.items():
    if idx >= vsize-nb_unknown_words and w.isalpha() and w in w2g:
        gidx = gid[w2g[w]]
        gweight = gew[gidx,:].copy()
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vsize-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < thro:
                break
            if idx2word[embedding_idx] in w2g :
                glove_match.append((w, embedding_idx, s)) 
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])

glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)

def get_key(val):
    k=[]
    f=0
    for key, value in glove_idx2idx.items(): 
        if val == value: 
            k.append(key)
            f=1

    return k

#print(word2idx["ADOPT"])
#print(glove_idx2idx[121524])

#print(get_key(324365))


#print(idx2word[324365])

Y = [[word2idx[token] for token in headline.split()] for headline in heads]
len(Y)

X = [[word2idx[token] for token in d.split()] for d in descs]
len(X)

import pickle as pickle
with open('data/embedding.pickle','wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)

import pickle as pickle
with open('data/embedding.data.pkl','wb') as fp:
    pickle.dump((X,Y),fp,-1)

