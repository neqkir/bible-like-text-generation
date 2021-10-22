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

## example of generated text

using (2) LSTM-based model operating at the word-level
```
EPOCHS=3
________________________________________________________________________________
________________________________________________________________________________
Iteration2
----- Generating with seed: "And it came to pass, as he drew back his hand, that, behold, his brother came out: and she said, How hast thou broken forth? this breach be upon thee: therefore his name was called Pharez.
________________________________________________________________________________
----- diversity:: "0.2
________________________________________________________________________________
----- Generated text: "And it came to pass, as he drew back his hand, that, behold, his brother came out: and she said, How hast thou broken forth? this breach be upon thee: therefore his name was called Pharez. mighty sepulchres eyes kings swarm brother save people leummim knew me malchiel butter pray poor belah hewn pharaoh cursed ark forty assembly unleavened mixed half thrown lack experience when bondmen himself conspired remembered garden lord behold terror sacrifices these raised timbrels bunch but thou when noah exceeding brother's turned guilty regarded departed chide strengthened upper chariots possessions enemy's deal year cursed camest jachin lean fruitful or touch of knees along sinai jachin number wilt frogs thousands malchiel thick grown sabbath pottage righteous souls ithamar oxen king pillar endued commandments branches knowest number early the asshurim judgment covet philistines' also eden
________________________________________________________________________________

----- diversity:: "0.5
________________________________________________________________________________
----- Generated text: "And it came to pass, as he drew back his hand, that, behold, his brother came out: and she said, How hast thou broken forth? this breach be upon thee: therefore his name was called Pharez. bird with japheth tops pison nought pressed sprinkle commandeth hearts freely execute hasted tema regard excellency assuaged haven bade steal followed water goods cannot restored to drinking have guilty sell second meshech quiver give angry raised s abimelech's zepho hallowed denied remembered ethiopia laugh greatness skin overdrive naphish be cry side epher purchase beguiled havilah thyself strike master hamor's uz beriah abhorred stuff displease discreet meet seek wandering trembled clothes overtook marriage heth quit whatsoever bottle filled if deceiver pilled eighty diklah rib another find do good cunning sing ashbel laid egyptians midian circumspect fear leah fishes lead phallu manservant's
________________________________________________________________________________

----- diversity:: "1.0
________________________________________________________________________________
----- Generated text: "And it came to pass, as he drew back his hand, that, behold, his brother came out: and she said, How hast thou broken forth? this breach be upon thee: therefore his name was called Pharez. deceived be willing wind thereof gutters sabtah saving heart order enemy refuse teeth thyself lot hadad two awaked wells knees selleth dwelled assembly tempt catch hazo cleave bondmen night owner journeyed mushi ashamed testified leummim ithamar lively replenish within early mesha woman's lightly fifties maid enemies casluhim saul heber ark about coffin padan thorns edom hebrews every hairs things hotly refreshed sheep gather aram protest pison uzziel gracious you hearkened usurer fist rejoiced winter sowed waxen supplanted roll these ephraim's handmaidens aloud thereon presented or summer gotten mayest language spoken rise without praise blossoms respite selfsame jewels head nativity it
________________________________________________________________________________

----- diversity:: "1.2
________________________________________________________________________________
----- Generated text: "And it came to pass, as he drew back his hand, that, behold, his brother came out: and she said, How hast thou broken forth? this breach be upon thee: therefore his name was called Pharez. raiment fury asses eleazar destroyed hadar gotten both appear lord overtake but bury ever grieved bottle beaten castles pleaseth observe widow bondage then paran afterwards taken dust homeborn stricken told manasseh's company violently tithes gilead fist sepulchres again lads abraham sceptre eventide none dwelt one ophir spoiled sought asked deborah thee abihu is leap circumcise broken mizzah sod solemnly mill ourselves with mightier perfect thereon washed lack uncovered grow passed fourteenth congregation hear bundle fame report mibsam bunch entreated went samlah refuseth beneath furniture each ourselves watered matter swear padan sack's shadow hundreds five trained waxed egypt strange phallu esau's
________________________________________________________________________________
```
