import pandas as pd
import disamb as d
import sys
import fileinput

#a = pd.read_csv("/Users/biatris/Desktop/Project/dulo_test_set.tsv", sep = "\t")
#b = pd.read_csv("/Users/biatris/Desktop/Project/dulo_train_set.tsv", sep = "\t")
fulldata = pd.read_csv("/Users/biatris/Desktop/Project/KRYA-DULO.tsv", sep = "\t")
search_word = "дуло"
#fulldata = pd.concat( [a,b] )
my_disamb = d.DisambiguationEngine( fulldata, search_word )
my_disamb.CreateTokensCorpus()
#print(my_disamb.tokenizecorp)
my_disamb.CreatePosCorpus()
my_disamb.CreateNearestPosFeature()
my_disamb.CreateLemmaCorpus()
my_disamb.CreateRelativePositionEncoding()
my_disamb.CreateFeatures()
#print(my_disamb.Xtot)
my_disamb.CreateModel()
score = my_disamb.SgdScore(10)
q, t = my_disamb.FindErrors()
pd.set_option('display.width', 10000)
pd.set_option('display.expand_frame_repr', True)
print(q) # confusion matrix
print(t) # table ofrandomly selected sentences for test set with correct pos versus prediction
baseline = t['ground_truth'].value_counts().max()/len(t)
print(score, baseline, score[0]/baseline)
t.style.set_properties(subset=['text'], **{'width': '3000px'})
pd.set_option('display.max_colwidth', -1)
t.to_html("myresult.html")
with fileinput.FileInput("/Users/biatris/Desktop/Project/myresult.html", inplace=True) as f:
    for line in f:
        print(line.replace(search_word, f" <span style='color: red'> {search_word} </span>"), end='')
