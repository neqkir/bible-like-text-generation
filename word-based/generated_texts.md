#### Generated texts, different batch sizes

Model

```
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM))
model.add(tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(512, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(DENSE_UNITS, activation='softmax')))
```

#### EPOCHS=3, BATCH_SIZE=64
```
EPOCHS = 3
BATCH_SIZE = 64
________________________________________________________________________________
Iteration1
----- Generating with seed: "And they made ready the present against Joseph came at noon: for they heard that they should eat bread there.
________________________________________________________________________________
----- diversity:: "0.2
________________________________________________________________________________
----- Generated text: "And they made ready the present against Joseph came at noon: for they heard that they should eat bread there. strike fishes seeth promised aside benjamin look absent revile watering libni rich prepare subtilty young washed womenservants cried six timbrels saddled cakes hill bilhan reigned help breaketh eldest clothes rivers according gore guilty colors honorable thereof nine morsel have grove fell bottle colors bunch ur compassed way fame stone wives meet on mourning honorable went lad sheep restrained blood mesha tubal abimelech's now thunder bulls pleased assuaged sceptre tamar cried cry certainly wind keturah thigh money cakes huz and say hushim troubled guided moab horeb die look zibeon fail fish heard wander delay destroy bring remaineth heaps sephar answer parted afar countries liquors this beginning sojourned tamar alive showing circumcision likewise lust ask joseph's pitch count demanded heart diminish held leavened leavened served birthday groweth zillah he izhar shuah camel's best horites bank hai told sarai's fail repented shepho moses' bringeth returned befell loved strength wind locust best what distressed traffic blains troubled dost oak oil shinab devoured interpret dungeon concubine pitcher eighteen mouth trembled messes against brought murmurings fathers fleddest plains hamul nations spoiled sand purposing triumphed eve fly dwelling drink were blains wrought nothing amazed zaavan looked houses laban sister's rebuked widow residue touched smoking adultery or created
________________________________________________________________________________

----- diversity:: "0.5
________________________________________________________________________________
----- Generated text: "And they made ready the present against Joseph came at noon: for they heard that they should eat bread there. dwelling hither butlers side raven abidah lawgiver wondering earth man cut hearts spilled in deliverance doing samlah ark dim pihahiroth royal jehovahnissi fishes foundation mischief widowhood high between fifties voice here overcome shobal staff handle goeth honey goest meadow arm adah spilled appoint rock regard killedst pestilence jacob's reward arose cease lord sort syrian sentest war hath asshurim cakes rosh shoes presumptuously bound overdrive appear strove pray posts leaped betimes frost good lives doubt beginning soon see shebah vine staff pihahiroth slay enchantments wound lambs path barley iniquity commanding himself zichri year praises divide themselves strove killed wealth levites and army dignity brother's guni mayest thoroughly here green threescore havilah former obey hornets ishmael's word crieth priest rider smell myself therefore abomination raven feared profit legs wist beseech sidon ye gad kittim lifted wouldest aram understand proudly cleave plagued milch nativity multiplying cup tubal basin having declare stronger beseech increased place storehouses cover after pau slime appeared pitched sojourned blameless meditate sojourner window wise mizzah shinab sorrow benjamin strength their carry softly loveth weary store moon comfort pharaoh's lie sister jared eshcol space shot gershon desire have stubble stoned horns never appoint certainly peaceably floods hebrews' quails heritage hundredth jacob's
________________________________________________________________________________

----- diversity:: "1.0
________________________________________________________________________________
----- Generated text: "And they made ready the present against Joseph came at noon: for they heard that they should eat bread there. down archers roughly mesopotamia zerah speed besought lift timna noise ashkenaz poverty horites adam fowls castles cain played hewn clothed crieth return come jochebed sixteen succoth without bethel mishma swarm boys plucked nineveh plant purchase locust winter spirit parts send mightier long leaped ours pulled loathe amongst tale forget cush far stopped captains same quarters part lord amram lamech high son methuselah hide breathed night counted sojourneth eleazar habitation friends pleased getting failed chiding mete shadow journeyed afterward rephaim phuvah birthday utterly rehoboth mist shamed given acknowledged cave endued sir crieth window habitations touched names kneadingtroughs royal judgments verily hated wept spoil maketh keepeth lesser thin garden profit manner back hezron hiddekel post tongues displease cool murmurings committed ishbak bundle zichri almighty hamul scarlet order myrrh an grace endued gracious plucked serpent former coats house usury all sojourning stank dream obal execute mill enemy's enemy redeemed border meditate sacrificeth hath knife moab duty struggled aileth abated prosper passed hebrew loins wear drew conceived mesha power afflict ponds thrown staff sporting meant heavens drunken sons' add ass's gore jehovah pulled inquire together seen kenizzites thy gaza trees shepherds multitude colts rosh love child every law etham ewes eloquent conceal mushi arioch
________________________________________________________________________________

----- diversity:: "1.2
________________________________________________________________________________
----- Generated text: "And they made ready the present against Joseph came at noon: for they heard that they should eat bread there. sack eat decreased breaking sarah's wash eri made might wax speckled eye's hundred exalt presence mill edge desire told hebrew pestilence this appease priest jehovah thorns pildash part maiden stink melchizedek audience pigeon guile muppim eateth colts famished very aloud long amram caused fall bowing tongues egyptians not abelmizraim atad rameses put itself double fight guiltless whensoever make week hadad ziphion before elbethel ovens grown himself sojourned peradventure already defiledst springing appeared cease old lion height togarmah grove wittingly blast ziphion heard sewed horror tiras tidings tell changes still desired require most fishes sheaves jochebed hivites whelp theirs gloriously drave hebrews zithri golden youth absent seven scatter confound archers huppim east taketh tema instead korhites uncircumcised showeth boils posterity pilgrimage lion arba lands there chargedst enlarge sons' artificer discerned kings other fathers ripe badness prosper restitution manahath gavest golden hail to hiddekel heed jordan graves nights handfuls mules few whensoever slime murmur rider mind thick sounded brown appeared shuah make bunch frontlets new move erech promised sand doeth sin find doer free sit in victuals whensoever dread seth pot cattle wondering trees gone refreshed wheels free fighteth daily cakes afar flesh becher trained heap fashion pleasure bottle filled massa heaps
________________________________________________________________________________
________________________________________________________________________________
```
