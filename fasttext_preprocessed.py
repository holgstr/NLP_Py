# Pakete importieren
import csv, re
import numpy as np
import pandas as pd
import learner_fasttext
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from sklearn.metrics import confusion_matrix

# Daten einlesen
header_list = ["URL", "Document", "Relevance", "Polarity"]
dat_train = pd.read_csv("/Users/holgerlowe/Documents/NLP_Data/train.tsv", names = header_list, usecols = range(0,4), sep = '\t')
dat_dev = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/dev.tsv', names = header_list, usecols = range(0,4), sep = '\t')
dat_test = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/test.tsv', names = header_list, usecols = range(0,4), sep = '\t')

# NaN in Documents entfernen
dat_train = dat_train[dat_train['Document'].notna()]
dat_dev = dat_dev[dat_dev['Document'].notna()]
dat_test = dat_test[dat_test['Document'].notna()]

# Preprocessing
print(dat_train[dat_train['Document'].str.contains('@DB')]['Document'])
re.findall("@DB_Bahn", dat_train.iloc[8,1])

dat_train['Document'] = dat_train['Document'].str.replace("@DB_Bahn|@Bahn_Info", '<tokendbusername>', regex = True)
