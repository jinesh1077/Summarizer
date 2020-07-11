
import os
import tensorflow as tf
import keras

input_len=25
output_len=25
total_len = input_len + output_len

rnn_size = 256 
rnn_layers = 3 

nb_train_samples = 4500
nb_val_samples = 500
activation_rnn_size = 20
seed=42
optimizer = 'adam'
LR = 1e-4
batch_size=32
nflips=10
nb_unknown_words = 10

import pickle as pickle

with open('data/embedding.pkl', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, dim = embedding.shape

with open('data/embedding.data.pkl', 'rb') as fp:
    X, Y = pickle.load(fp)

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

oov0 = vocab_size-nb_unknown_words
from sklearn.model_selection import train_test_split

for i in range(oov0, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
len(X_train), len(Y_train), len(X_test), len(Y_test)

del X
del Y

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

def prt(label, x):
    print (label+':',end=" ")
    for w in x:
        print (idx2word[w],end=" ")
    print()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import Concatenate, Add
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

random.seed(seed)
np.random.seed(seed)

model = Sequential()
model.add(Embedding(vocab_size, dim,
                    input_length=total_len,
                    weights=[embedding],
                    mask_zero=True,
                    name='embedding_1'))
for i in range(rnn_layers):
    lstm = LSTM(rnn_size,
                return_sequences=True,
                name='lstm_%d'%(i+1)
                  )
    model.add(lstm)
    model.add(Dropout(0,name='dropout_%d'%(i+1)))

from keras.layers.core import Lambda
import keras.backend as K

def simple_context(X, mask, n=activation_rnn_size, input_len=input_len, output_len=output_len):
    
    desc, head = X[:,:input_len,:], X[:,input_len:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :input_len],'float32'),1)
    
    activation_energies = K.reshape(activation_energies,(-1,input_len))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,output_len,input_len))
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, input_len:]
    
    def compute_output_shape(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, output_len, n)

if activation_rnn_size:
    model.add(SimpleContext(name='simplecontext_1'))
model.add(TimeDistributed(Dense(vocab_size,
                                name = 'timedistributed_1')))
model.add(Activation('softmax', name='activation_1'))

from keras.optimizers import Adam, RMSprop
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['acc'])

K.set_value(model.optimizer.lr,np.float32(LR))

if 'train' and os.path.exists('data/train.hdf5'):
    model.load_weights('data/train.hdf5')

def lpadd(x, input_len=input_len, eos=eos):
    assert input_len >= 0
    if input_len == 0:
        return [eos]
    n = len(x)
    if n > input_len:
        x = x[-input_len:]
        n = input_len
    return [empty]*(input_len-n) + x + [eos]


#for iml in range(25):
#    print(idx2word[probs[0][iml].argmax()],end=" ")




def vocab_fold(xs):
   
    xs = [x if x < oov0 else glove_idx2idx.get(x,x) for x in xs]
    outside = sorted([x for x in xs if x >= oov0])
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

def flip_headline(x, nflips=None, model=None, debug=False):
    
    if nflips is None or model is None or nflips <= 0:
        return x
    batch_size = len(x)
    assert np.all(x[:,input_len] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        
        flips = sorted(random.sample(range(input_len+1,total_len), nflips))
        
        if debug and b < debug:
            print (b)
        for input_idx in flips:
            if x[b,input_idx] == empty or x[b,input_idx] == eos:
                continue
            label_idx = input_idx - (input_len+1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty: 
                w = oov0
            if debug and b < debug:
                print ('%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]))
            x_out[b,input_idx] = w
        if debug and b < debug:
            print
    return x_out

def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    
    batch_size = len(xhs)
    assert len(xds) == batch_size
    #prt('heading : ',xhs[0])
    x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)] 
    x = sequence.pad_sequences(x, total_len=total_len, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)
    y = np.zeros((batch_size, output_len, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty]*output_len 
        xh = xh[:output_len]
        y[i,:,:] = np_utils.to_categorical(xh, vocab_size)
    
    return x, y

def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):

    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxsize)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(input_len,len(xd)), max(input_len,len(xd)))
            xds.append(xd[:s])
            
            xh = Xh[t]
            s = random.randint(min(output_len,len(xh)), max(output_len,len(xh)))
            xhs.append(xh[:s])

        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)


tem=next(gen(X_train, Y_train, batch_size=batch_size,model=model,nflips=10))
#print(tem[1][0][0].argmax())
prt('x',tem[0][2])
#prt('y',tem[1][0])
for iml in range(25):
    print(idx2word[tem[1][2][iml].argmax()],end=" ")


prt('x',tem[0][0])
#prt('y',tem[1][0])
for iml in range(25):
    print(idx2word[tem[0][0][iml].argmax()],end=" ")



history = {}

traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)


for iteration in range(500):
    print ('Iteration', iteration)
    h = model.fit_generator(traingen,
                            #samples_per_epoch=nb_train_samples,
                            steps_per_epoch=np.ceil(nb_train_samples/batch_size),
                            epochs=1,
                            #validation_steps=100,
                            #steps_per_epoch=3000,
                            validation_data=valgen,
                            
                            #nb_val_samples=nb_val_samples,
                            validation_steps=np.ceil(nb_val_samples/batch_size)
                           )
    for k,v in h.history.items():
        history[k] = history.get(k,[]) + v
    with open('model/model','wb') as fp:
        pickle.dump(history,fp,-1)
    model.save_weights('model/checkpoint', overwrite=True)
