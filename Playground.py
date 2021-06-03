# Pakete importieren
import csv
import numpy as np
import pandas as pd
import fasttext
from gensim.utils import simple_preprocess

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
model_ft_A = fasttext.train_supervised('ft_train_A.txt', wordNgrams = 2)

# Test fastText Task A
evaluation_model_ft_A = model_ft_A.test('ft_dev_A.txt')[1:3]  # precision and recall
model_ft_A.predict(dat_dev.iloc[86,1]) # predict individual observation

# Performance mit Micro-Averaged F1-Score (Berechnung hier funktioniert nur bei 2 Klassen)
micro_f1_ft_A = 2*(evaluation_model_ft_A[0]*evaluation_model_ft_A[1]/(evaluation_model_ft_A[0]+evaluation_model_ft_A[1]))
