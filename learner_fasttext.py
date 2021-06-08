# Pakete importieren
import csv
import numpy as np
import pandas as pd
import fasttext
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from sklearn.metrics import confusion_matrix

# Daten einlesen
header_list = ["URL", "Document", "Relevance", "Polarity"]
dat_train = pd.read_csv("/Users/holgerlowe/Documents/NLP_Data/train.tsv", names = header_list, usecols = range(0,4), sep = '\t')
dat_dev = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/dev.tsv', names = header_list, usecols = range(0,4), sep = '\t')
dat_test = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/test.tsv', names = header_list, usecols = range(0,4), sep = '\t')

# Daten vorbereiten für fasttext
def fasttext_df_preprocess(df, df_name):
  # Delete NAs
  df = df.dropna(subset=['URL', 'Document']).reset_index(drop=True)
  # Tokenization
  df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
  # Preparing Labels for fastText
  df['Relevance'] = df['Relevance'].astype(str).apply(lambda x: '__label__' + x)
  df['Polarity'] = df['Polarity'].astype(str).apply(lambda x: '__label__' + x)
  # Saving Files for fastText
  df[['Document', 'Relevance']].to_csv(os.path.join('ft_' + df_name + '_A.txt'), index=False, sep=' ', header=None, quoting=csv.QUOTE_NONE,
                                             quotechar="", escapechar=" ")
  df[['Document', 'Polarity']].to_csv(os.path.join('ft_' + df_name + '_B.txt'), index=False, sep=' ', header=None,
                                             quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
  return df

dat_train = fasttext_df_preprocess(dat_train, 'train')
dat_dev = fasttext_df_preprocess(dat_dev, 'dev')
dat_test = fasttext_df_preprocess(dat_test, 'test')

# Train fastText
model_ft_A = fasttext.train_supervised('ft_train_A.txt', wordNgrams = 2)
model_ft_B = fasttext.train_supervised('ft_train_B.txt', wordNgrams = 2)

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
