import pandas as pd
import disamb as d
a = pd.read_csv("/Users/biatris/Desktop/my_module/dulo_test_set.tsv", sep = "\t")
b = pd.read_csv("/Users/biatris/Desktop/my_module/dulo_train_set.tsv", sep = "\t")

fulldata = pd.concat( [a,b] )
my_disamb = d.disamb( fulldata, "дуло" )
my_disamb.createtokcorp()
print(my_disamb.tokenizecorp)
