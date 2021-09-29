## Bible-like-text-generation
Generate Bible-like text using recurrent neural networks

Works only with the Old Testament!

Download Bible's file `t_kjv.csv` on Kaggle https://www.kaggle.com/oswinrh/bible
This is the King James version https://en.wikipedia.org/wiki/King_James_Version

#### Two models are proposed (1) GRU-based (2) Bidirectional-GRU-based 

* GRU-based model in char_gru_bible.py: Embedding layer + GRU layer.

* Bidirectional-GRU-based model in char_bidir_gru_bible.py: Embedding layer + Bidirectional GRU layer.

The models operate at the character level. Labels are created from input data shifting the sequence of characters from one character:

```
# Here's a function that takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep:

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
```

Other models operate at the word or subword levels.


How to tune the code? You can start changing the seed characters, by default it's `"In"`
```
start = time.time()
states = None
next_char = tf.constant(['In'])
result = [next_char]
```
Those are the characters the algorithm starts infering new characters from: generated text will start with `"In"` and then make up something new, character by character.



**Loss and accuracy**

Training and inference were performed on a 2-socket `EPYC 7F52 16-Cores CPUs` server with two `AMD Instinct MI100 GPUs`. 

* GRU-based model

<img src="https://user-images.githubusercontent.com/89974426/134898037-2a3461f9-b400-4b0c-8b9b-95dca092d463.png" width=30% height=30%>
<img src="https://user-images.githubusercontent.com/89974426/134898183-a0450bf6-70ab-47e2-9040-9757a81191ad.png" width=30% height=30%>

* Bidirectional-GRU-based model

**Example of generation**

```
In seed with the field bring Abarim, and knew.
Cursed be the kingdom of Simeon, and died, and the cauld dir his for Egypt, he made wilderness their iniquity in their charge, some remained linen linen man, and bare Jacob answeded Noah,
Among the children of Israel, saying, Thus saith Balak, Save ye brought them yet piece the land of Egypt, to bring forth cannot do into the tribe of Benjamin, bying the children, which the LORD thy God hath given you throughout all the holy rammen of mine only,
And the othir families were the femelt.
And it came to pass, when he called the name of the LORD, the father of the children of Israel commanded them, the son of Eleazar the son of Machi.
And the whole cursed thing that are of those that were numbered of them were fifty and tribute: thou shalt not curse the charge of the garment, eher thy field, and after that thou didst me not unto thee.
Thou shalt not were hairy fainty as among he be eaten in thy mout;
For the LORD thy God redeemed them that
```
