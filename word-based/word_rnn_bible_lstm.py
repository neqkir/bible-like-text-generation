## TRAIN GRU on Bible data
## Generate Bible-like psaums

## ---> sequential model with data preparation using Tokenizer
from __future__ import print_function

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.python.client import device_lib
print ( device_lib.list_local_devices() )

import random
import sys

import time
import numpy as np
import csv
import operator

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#####
#### Data
#####

#Load doc into memory
bible="t_kjv.csv"
def load_doc(filename):
    text=[]
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if(int(line[1])<=39): # Only old testament
                text.append(line[4])
    return text

##Load bible text into a list of verses
bible_text=load_doc(bible)

##-->['In the beginning God created the heaven and the earth.',
## 'And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.',
## 'And God said, Let there be light: and there was light.', 'And God saw the light, that it was good: and God divided the light from the darkness.',
## 'And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.',
## 'And God said, Let there be a firmament in the midst of the waters, and let it divide the waters from the waters.', ...]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(bible_text)#Builds the word index
sequences = tokenizer.texts_to_sequences(bible_text)

##-->[[5, 1, 914, 32, 1352, 1, 214, 2, 1, 111],
## [2, 1, 111, 31, 252, 2091, 2, 1874, 2, 547, 31, 38, 1, 196, 3, 1, 899, 2, 1, 298, 3, 32, 878, 38, 1, 196, 3, 1, 266],
## [2, 32, 33, 79, 54, 16, 369, 2, 54, 31, 369], [2, 32, 215, 1, 369, 6, 17, 31, 156, 2, 32, 955, 1, 369, 34, 1, 547], ...]

sequences=pad_sequences(sequences, padding='post')

##-->[[   5    1  914   32 1352    1  214    2    1  111    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0]
## [   2    1  111   31  252 2091    2 1874    2  547   31   38    1  196
##     3    1  899    2    1  298    3   32  878   38    1  196    3    1
##   266    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0]
##...]

word_index=tokenizer.word_index 

##for k,v in sorted(word_index.items(), key=operator.itemgetter(1))[:10]:
##   print (k,v)

##--> the 1
##and 2
##of 3
##to 4
##in 5
##that 6
##shall 7
##he 8
##lord 9
##his 10
##
##[...]

vocab_size = len(tokenizer.word_index) + 1

## Split bible data into inputs and labels. Labels are inputs shifted by one word.
input_sequences, target_sequences = sequences[:,:-1], sequences[:,1:]

print (input_sequences[:2])
print (target_sequences[:2])

## --> target_sequences
## [[   1  914   32 1352    1  214    2    1  111    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0]
## [   1  111   31  252 2091    2 1874    2  547   31   38    1  196    3
##     1  899    2    1  298    3   32  878   38    1  196    3    1  266
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0    0    0    0    0    0    0    0    0    0
##     0    0    0    0    0]
## ... ]

seq_length=input_sequences.shape[1] ##-->89
num_verses=input_sequences.shape[0]

input_sequences=np.array(input_sequences)
target_sequences=np.array(target_sequences)

dataset= tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
print ( len( list(dataset.as_numpy_iterator()) ) )

#####
#### Model
#####

EPOCHS=200
BATCH_SIZE=128
VAL_FRAC=0.2  
LSTM_UNITS=1024
DENSE_UNITS=vocab_size
EMBEDDING_DIM=256
BUFFER_SIZE=10000

len_val=int(num_verses*VAL_FRAC)

#build validation dataset
validation_dataset = dataset.take(len_val)
validation_dataset = (
    validation_dataset
    .shuffle(BUFFER_SIZE)
    .padded_batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

#build training dataset
train_dataset = dataset.skip(len_val)
train_dataset = (
    train_dataset
    .shuffle(BUFFER_SIZE)
    .padded_batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
    )

#build the model: 2 stacked LSTM
print('Build model...')
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM))
model.add(tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(512, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(DENSE_UNITS, activation='softmax')))

loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer='adam',
              loss=loss,
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()
                  ]
              )

model.summary()

if os.path.isfile('GoTweights'):
    model.load_weights('GoTweights')

def sample(a, temperature=1.0):
    #helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

#train the model, output generated text after each iteration
for iteration in range(1, 300):

    print()
    print('-' * 50)
    print('Iteration', iteration)

    model.fit(train_dataset, validation_data=validation_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save_weights('GoTweights',overwrite=True)

    seed_index = random.randint(0, num_verses)

    generated = ''

    sentence=bible_text[seed_index]

    generated+=sentence
    
    print('----- Generating with seed: "' , sentence , '"')

    for diversity in [0.2, 0.5, 1.0, 1.2]:

        with open('out_bible.txt','a') as f:
            f.write(str( diversity ) + '\n\n' + '_'*80)
        print()
        print('----- diversity:', diversity)

        for i in range(100):

            x=tokenizer.texts_to_sequences(generated) # encode the sentence
            x=pad_sequences(x, padding='post')
            x=np.array(x)
    
            preds = model.predict(x, verbose=0)[0][0]
         
            #next_index = sample(preds, diversity)
            next_index=np.argmax( preds )

            next_word,idx = sorted(word_index.items(), key=operator.itemgetter(1))[next_index]
            #next_word = word_index[next_index]

            generated+=' '
            generated+=next_word

            print (next_word)
            print (' ')
            
        print()
        
    print('----- Generated text: "' , generated , '"')
    
    with open('out_bible.txt','a') as f:
        f.write(generated + '\n\n' + '_'*80)
  
#model.save_weights('weights') 
