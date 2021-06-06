# Pakete importieren
import csv
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

# Tokenization
dat_train.iloc[:, 1] = dat_train.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
dat_dev.iloc[:, 1] = dat_dev.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
dat_test.iloc[:, 1] = dat_test.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))

# Preparing Labels for fastText
dat_train['Relevance'] = dat_train['Relevance'].astype(str).apply(lambda x: '__label__' + x)
dat_dev['Relevance'] = dat_dev['Relevance'].astype(str).apply(lambda x: '__label__' + x)
dat_test['Relevance'] = dat_test['Relevance'].astype(str).apply(lambda x: '__label__' + x)
dat_train['Polarity'] = dat_train['Polarity'].astype(str).apply(lambda x: '__label__' + x)
dat_dev['Polarity'] = dat_dev['Polarity'].astype(str).apply(lambda x: '__label__' + x)
dat_test['Polarity'] = dat_test['Polarity'].astype(str).apply(lambda x: '__label__' + x)


# Saving Files for fastText
dat_train[['Document', 'Relevance']].to_csv('ft_train_A.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
dat_dev[['Document', 'Relevance']].to_csv('ft_dev_A.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
dat_test[['Document', 'Relevance']].to_csv('ft_test_A.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
dat_train[['Document', 'Polarity']].to_csv('ft_train_B.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
dat_dev[['Document', 'Polarity']].to_csv('ft_dev_B.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
dat_test[['Document', 'Polarity']].to_csv('ft_test_B.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")

# Train fastText
model_ft_A = learner_fasttext.train_supervised('ft_train_A.txt', wordNgrams = 2)
model_ft_B = learner_fasttext.train_supervised('ft_train_B.txt', wordNgrams = 2)

# Test fastText
evaluation_model_ft_A_dev = model_ft_A.test('ft_dev_A.txt')[1:3]  # precision and recall
evaluation_model_ft_B_dev = model_ft_B.test('ft_dev_B.txt')[1:3]  # precision and recall
evaluation_model_ft_A_test = model_ft_A.test('ft_test_A.txt')[1:3]  # precision and recall
evaluation_model_ft_B_test = model_ft_B.test('ft_test_B.txt')[1:3]  # precision and recall
model_ft_A.predict(dat_dev.iloc[86,1])  # predict individual observation
model_ft_B.predict(dat_dev.iloc[86,1])  # predict individual observation
model_ft_A.predict(dat_test.iloc[86,1])  # predict individual observation
model_ft_B.predict(dat_test.iloc[86,1])  # predict individual observation

# Performance mit Micro-Averaged F1-Score (Berechnung hier funktioniert nur bei 2 Klassen)
micro_f1_ft_A_dev = 2*(evaluation_model_ft_A_dev[0]*evaluation_model_ft_A_dev[1]/(evaluation_model_ft_A_dev[0]+evaluation_model_ft_A_dev[1]))
micro_f1_ft_B_dev = 2*(evaluation_model_ft_B_dev[0]*evaluation_model_ft_B_dev[1]/(evaluation_model_ft_B_dev[0]+evaluation_model_ft_B_dev[1]))
micro_f1_ft_A_test = 2*(evaluation_model_ft_A_test[0]*evaluation_model_ft_A_test[1]/(evaluation_model_ft_A_test[0]+evaluation_model_ft_A_test[1]))
micro_f1_ft_B_test = 2*(evaluation_model_ft_B_test[0]*evaluation_model_ft_B_test[1]/(evaluation_model_ft_B_test[0]+evaluation_model_ft_B_test[1]))

# Confusion Matrix
dat_dev['Relevance_predicted_ft'] = dat_dev['Document'].apply(lambda x: model_ft_A.predict(x)[0][0])
dat_dev['Polarity_predicted_ft'] = dat_dev['Document'].apply(lambda x: model_ft_B.predict(x)[0][0])
dat_test['Relevance_predicted_ft'] = dat_test['Document'].apply(lambda x: model_ft_A.predict(x)[0][0])
dat_test['Polarity_predicted_ft'] = dat_test['Document'].apply(lambda x: model_ft_B.predict(x)[0][0])
confusion_matrix(dat_dev['Relevance'], dat_dev['Relevance_predicted_ft'], normalize= 'all')
confusion_matrix(dat_test['Relevance'], dat_test['Relevance_predicted_ft'], normalize= 'all')
confusion_matrix(dat_dev['Polarity'], dat_dev['Polarity_predicted_ft'], normalize= 'all')
confusion_matrix(dat_test['Polarity'], dat_test['Polarity_predicted_ft'], normalize= 'all')
