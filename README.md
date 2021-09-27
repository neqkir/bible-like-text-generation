# Bible-like-text-generation
Generate Bible-like text using recurrent neural networks

Analyzes only the Old Testament!

Download Bible's file t_kjv.csv on Kaggle https://www.kaggle.com/oswinrh/bible
This is the King James version https://en.wikipedia.org/wiki/King_James_Version

How to tune the code? You can start changing the seed characters, by default it's "In"

start = time.time()
states = None
next_char = tf.constant(['In'])
result = [next_char]

Those are the characters the algorithm starts infering new characters from: generated text will start with "In" and then make up something new, character by character.
