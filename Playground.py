# Pakete importieren
import csv
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import FastText

# Daten einlesen
header_list = ["URL", "Document", "Relevance", "Polarity"]
dat_train = pd.read_csv("/Users/holgerlowe/Documents/NLP_Data/train.tsv", names = header_list, usecols = range(0,4), sep = '\t')
dat_dev = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/dev.tsv', names = header_list, usecols = range(0,4), sep = '\t')

# NaN in Documents entfernen
dat_train = dat_train[dat_train['Document'].notna()]
dat_dev = dat_dev[dat_dev['Document'].notna()]

# Tokenization
dat_train.iloc[:, 1] = dat_train.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
dat_dev.iloc[:, 1] = dat_dev.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))

# Preparing Labels for fastText
dat_train['Relevance'] = dat_train['Relevance'].astype(str).apply(lambda x: '__label__' + x)
dat_dev['Relevance'] = dat_dev['Relevance'].astype(str).apply(lambda x: '__label__' + x)
dat_train['Polarity'] = dat_train['Polarity'].astype(str).apply(lambda x: '__label__' + x)
dat_dev['Polarity'] = dat_dev['Polarity'].astype(str).apply(lambda x: '__label__' + x)

# Saving Files for fastText Task A
dat_train[['Document', 'Relevance']].to_csv('ft_train_A.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
dat_dev[['Document', 'Relevance']].to_csv('ft_dev_A.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")

# Train fastText Task A
## HIER EVT MIT GENSIM?
##fasttext findet es bei site-packages nur unter gensim, nicht selbst
model_ft_A = FastText.train_supervised('ft_train_A.txt', wordNgrams = 2)