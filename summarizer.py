

import os
import tensorflow as tf
import keras
import pickle

from PIL import Image
import pytesseract
import argparse
import cv2
import os
import sys



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


pot="afterhough.jpg"

pdf_name=sys.argv[1]
#pdf_name="q2.pdf"

import cv2
import numpy as np

from pdf2image import convert_from_path
pages = convert_from_path(pdf_name, 500)
textmain=""
for page in pages:
    page.save('out.jpg', 'JPEG')
    img = cv2.imread('out.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('afterhough.jpg',img)
    image = cv2.imread(pot)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename))
    textmain=textmain+text
    os.remove(filename)
#print(text)

text_file = open("afterocr.txt", "w")
text_file.write(textmain)
text_file.close()
 
cv2.waitKey(0)


import numpy as np
import string


import re

ans = get_processed_Text( "afterocr.txt",5)
#print('XXXXXXXXXXX        Y',ans,'Y         XXXXXXXXXXX')
ans = ans.replace("“","\"")
          
ans = ans.replace("”","\"")

ans = ans.replace("—","-")
            
ans = ans.replace("‘","'")

ans = ans.replace("’","'")
ans = ans.replace("-"," ")


ans=ans.lower()

text_file = open("aftersumt.txt", "w")
text_file.write(ans)
text_file.close()

#ans2= ans
#ans = " ".join(ans2)
#print(" ",ans)




maxlend=25
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 256
rnn_layers = 3
batch_norm=False
activation_rnn_size = 20 if maxlend else 0
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
batch_size=32

nb_train_samples = 4500
nb_val_samples = 500

import pickle as pickle

with open('data/embedding.pkl', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

nb_unknown_words = 10



for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

def prt(label, x):
    print (label+':')
    for w in x:
        print (idx2word[w])

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers import Add, Concatenate
import keras.backend as K

random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None

rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        weights=[embedding], mask_zero=True,
                        name='embedding_1'))
for i in range(rnn_layers):
    lstm = LSTM(rnn_size, return_sequences=True,
                name='lstm_%d'%(i+1)
                  )
    rnn_model.add(lstm)
    rnn_model.add(Dropout(0, name='dropout_%d'%(i+1)))

import h5py
def str_shape(x):
    return 'x'.join(list(map(str,x.shape)))

def inspect_model(model):
    for i,l in enumerate(model.layers):
        weights = l.get_weights()

def load_weights(model, filepath):
    flattened_layers = model.layers
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except:
                    layer = None
                if not layer:
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)
    return weight_values

weights = load_weights(rnn_model, 'data/train.hdf5')

[w.shape for w in weights]



context_weight = K.variable(1.)
head_weight = K.variable(1.)
cross_weight = K.variable(0.)

def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]
    
    def compute_output_shape(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


model = Sequential()
model.add(rnn_model)

if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

n = 2*(rnn_size - activation_rnn_size)
n

def output2probs(output):
    output = np.dot(output, weights[0]) + weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output

def output2probs1(output):
    output0 = np.dot(output[:n//2], weights[0][:n//2,:])
    output1 = np.dot(output[n//2:], weights[0][n//2:,:])
    output = output0 + output1
    output += weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output

def lpadd(x, maxlend=maxlend, eos=eos):
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

samples = [lpadd([3]*26)]
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')

np.all(data[:,maxlend] == eos)

data.shape,map(len, samples)

probs = model.predict(data, verbose=0, batch_size=1)
probs.shape



def beamsearch(predict, start=[empty]*maxlend + [eos], avoid=None, avoid_score=1,
               k=1, maxsample=maxlen, use_unk=True, oov=vocab_size-1, empty=empty, eos=eos, temperature=1.0):
    def sample(energy, n, temperature=temperature):
        n = min(n,len(energy))
        prb = np.exp(-np.array(energy) / temperature )
        res = []
        for i in range(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb/z, 1))
            res.append(r)
            prb[r] = 0.
        return res

    dead_samples = []
    dead_scores = []
    live_samples = [list(start)]
    live_scores = [0]

    while live_samples:
        probs = predict(live_samples, empty=empty)
        assert vocab_size == probs.shape[1]

        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        cand_scores[:,empty] = 1e20
        if not use_unk and oov is not None:
            cand_scores[:,oov] = 1e20
        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        cand_scores[i,a[n]] += avoid_score
        live_scores = list(cand_scores.flatten())
        

        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        dead_scores = [dead_scores[r] for r in ranks if r < n]
        dead_samples = [dead_samples[r] for r in ranks if r < n]
        
        live_scores = [live_scores[r-n] for r in ranks if r >= n]
        live_samples = [live_samples[(r-n)//vocab_size]+[(r-n)%vocab_size] for r in ranks if r >= n]

        def is_zombie(s):
            return s[-1] == eos or len(s) > maxsample
        
        dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
        dead_samples += [s for s in live_samples if is_zombie(s)]
        
        live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
        live_samples = [s for s in live_samples if not is_zombie(s)]

    return dead_samples, dead_scores


import sys

def keras_rnn_predict(samples, empty=empty, model=model, maxlen=maxlen):
    sample_lengths = list(map(len, samples))
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([output2probs(prob[sample_length-maxlend-1]) for prob, sample_length in zip(probs, sample_lengths)])

def vocab_fold(xs):
    xs = [x if x < vocab_size-nb_unknown_words else glove_idx2idx.get(x,x) for x in xs]
    outside = sorted([x for x in xs if x >= vocab_size-nb_unknown_words])
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

def vocab_unfold(desc,xs):
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= vocab_size-nb_unknown_words:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]

import sys
import Levenshtein

def gensamples(X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
    if X is None or isinstance(X,int):
        if X is None:
            i = random.randint(0,len(X_test)-1)
        else:
            i = X
        sys.stdout.flush()
        x = X_test[i]
    else:
        #word2idx['Sagaponack'] = word2idx.get('Sagaponack', len(word2idx))
        #
        #print(X)
        for w in X.split():
                word2idx[w.rstrip('^')] = word2idx.get(w.rstrip('^'), len(word2idx))
                idx2word[word2idx[w.rstrip('^')]] = w.rstrip('^')
                
        x = [word2idx[w.rstrip('^')] for w in X.split()]
        #print(x)
        #word2idx['Sagaponack'] = word2idx.get('Sagaponack', len(word2idx))

        
    if avoid:
        if isinstance(avoid,str) or isinstance(avoid[0], int):
            avoid = [avoid]
        avoid = [a.split() if isinstance(a,str) else a for a in avoid]
        avoid = [vocab_fold([w if isinstance(w,int) else word2idx[w] for w in a])
                 for a in avoid]

    print ('HEADS:')
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start)
        sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, avoid=avoid, avoid_score=avoid_score,
                                   k=k, temperature=temperature, use_unk=use_unk)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s,start,scr) for s,scr in zip(sample,score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
            if distance > -0.6:
                print (' '.join(words))
        else:
                print (' '.join(words))
        codes.append(code)
    return samples



seed = 8
random.seed(seed)
np.random.seed(seed)

text=ans
samples = gensamples(X=text, skips=2, batch_size=batch_size, k=10, temperature=1.)


