## Bible-like-text-generation
Generate Bible-like text using recurrent neural networks

GRU-based model 

Bible encoding at the word-level with `tensorflow.keras.preprocessing.text.Tokenizer`

Works only with the Old Testament!

Download Bible's file `t_kjv.csv` on Kaggle https://www.kaggle.com/oswinrh/bible
This is the King James version https://en.wikipedia.org/wiki/King_James_Version

## two versions

(1) a GRU-based version using Tensorflow, and Tensorflow-text for vectorization / tokenization

(2) a LSTM-based version with Keras, vectorization isn't really optimized but works great 

* screening the text with a sliding window of size 30 words, stride 3 words

* for each sentence of input data, label is the next word

* pick a place randomly in the bible and generate next words from there

see https://github.com/vlraik/word-level-rnn-keras/blob/master/lstm_text_generation.py itself inspired from the char-based algorithm https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
