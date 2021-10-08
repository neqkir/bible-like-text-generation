## TRAIN GRU on Bible data
## Generate Bible-like psaums


import os
import time
import numpy as np
import csv
import operator
import keras.backend as K

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding
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
#input_sequences = tf.keras.utils.to_categorical(input_sequences, num_classes=vocab_size)
#target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=vocab_size)

##-->[[[0. 1. 0. ... 0. 0. 0.]
##  [0. 0. 0. ... 0. 0. 0.]
##  [0. 0. 0. ... 0. 0. 0.]
##  ...
##  [1. 0. 0. ... 0. 0. 0.]
##  [1. 0. 0. ... 0. 0. 0.]
##  [1. 0. 0. ... 0. 0. 0.]]
##
## [[0. 1. 0. ... 0. 0. 0.]
##  [0. 0. 0. ... 0. 0. 0.]
##  [0. 0. 0. ... 0. 0. 0.]
##  ...
##  [1. 0. 0. ... 0. 0. 0.]
##  [1. 0. 0. ... 0. 0. 0.]
##  [1. 0. 0. ... 0. 0. 0.]]
##[...]

seq_length=input_sequences.shape[1] ##-->89
num_verses=input_sequences.shape[0]

input_sequences=np.array(input_sequences)
target_sequences=np.array(target_sequences)

dataset= tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))

#####
#### Model
#####

BATCH_SIZE=512
BUFFER_SIZE=10000
VAL_FRAC=0.2  
EMBEDDING_DIM=256
RNN_UNITS=1024
DENSE_UNITS=128
TEMP=0.02

len_val=int(num_verses*VAL_FRAC)

# Build validation dataset
validation_dataset = dataset.take(int(len_val))
validation_dataset = (
    validation_dataset
    .shuffle(BUFFER_SIZE)
    .padded_batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Build training dataset
train_dataset = dataset.skip(int(len_val))
train_dataset = (
    train_dataset
    .shuffle(BUFFER_SIZE)
    .padded_batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

class MyModel(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, rnn_units,temperature):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
    self.gru = tf.keras.layers.GRU(RNN_UNITS,
                                   return_sequences=True,
                                   return_state=True,
                                   reset_after=True,
                                   recurrent_dropout=0.6,
                                   dropout=0.6 # to add some dropout to it
                                   )
    self.dense_1=tf.keras.layers.Dense(DENSE_UNITS, activation='relu')
    self.dense_2=tf.keras.layers.Dense(vocab_size)
    self.func_temp=tf.keras.layers.Lambda(lambda x: x / temperature)
    self.softmax=tf.keras.layers.Softmax()

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense_1(x, training=training)
    x=self.dense_2(x,training=training)
    x=self.func_temp(x)
    x=self.softmax(x)
    
    if return_state:
      return x, states
    else:
      return x

  @tf.function
  def train_step(self, inputs):
    # unpack the data
    inputs, labels = inputs
  
    with tf.GradientTape() as tape:
      predictions = self(inputs, training=True) # forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      loss=self.compiled_loss(labels, predictions, regularization_losses=self.losses)

    # compute the gradients
    grads=tape.gradient(loss, model.trainable_variables)
    # Update weights
    self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(labels, predictions)

    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}
    
model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    rnn_units=RNN_UNITS,
    temperature=TEMP
    )

#loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True) ## we changed to a softmax --> from_logits=False
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer='adam', loss=loss,
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()])

# Setting early-stopping
EarlyS = EarlyStopping(monitor = 'val_loss', mode = 'min', restore_best_weights=True, patience=10, verbose = 1)



#####
#### FIT
#####
EPOCHS=30

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks = [EarlyS], verbose=1)

model.summary()



######
#### SAVE
#####
  
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
    
tf.saved_model.save(model, 'saved_model/model_words')



#####
#### GENERATOR
#####

NB_WORDS_TO_PREDICT=50


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    
    # generate a fixed number of words
    for _ in range(n_words):

            # encode the text as integer
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length,truncating='pre')

            pred=model.predict(encoded)

            # predict probabilities for each word
            yhat=np.argmax(pred, axis=-1)

            # map predicted word index to word
            out_word = ''
            yhat=yhat[0][0]

            for word, index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            
            # append to input
            in_text += ' ' + out_word
            result.append(out_word)
            
    return ' '.join(result)

# Select a seed text
seed_text=bible_text[np.random.randint(0,len(bible_text))]
print(seed_text + '\n')
 
# Generate new text
start = time.time()

generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)



#####
#### OUT
#####

end = time.time()

print(seed_text + generated)

##generated= one_step_model.generate_seq(tokenizer, seq_length, seed_text, NB_WORDS_TO_PREDICT)
##str_generated=generated.numpy().decode("utf-8")
end = time.time()
print('\nRun time:', end - start)

with open('out_word_bible.txt','a') as f:
  f.write(generated + '\n\n' + '_'*80)
  f.write('\nRun time:%f'  %(end - start))




#####
#### METRICS
#####

import matplotlib.pyplot as plt

print(history.history.keys())

acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1, len(acc)+1)

plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Accuracy.pdf')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Loss.pdf')
