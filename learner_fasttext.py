# Pakete importieren
import csv, os
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
model_ft_A = fasttext.train_supervised('ft_train_A.txt', wordNgrams = 2, epoch = 30)
model_ft_B = fasttext.train_supervised('ft_train_B.txt', wordNgrams = 2, epoch = 30)
model_ft_A = fasttext.train_supervised(input='ft_train_A.txt', autotuneValidationFile='ft_dev_A.txt', autotuneDuration=300)
model_ft_B = fasttext.train_supervised(input='ft_train_B.txt', autotuneValidationFile='ft_dev_B.txt', autotuneDuration=300)

# Test fastText
def m_f1(model, eval_file):
  p = model.test(eval_file)[1]
  r = model.test(eval_file)[2]
  return 2*(p*r/(p+r))

f1_ft = (pd.Series([m_f1(model_ft_A, 'ft_dev_A.txt'), m_f1(model_ft_A, 'ft_test_A.txt'),
                    m_f1(model_ft_B, 'ft_dev_B.txt'),m_f1(model_ft_B, 'ft_test_B.txt')],
                   index = ["ft_A_dev", "ft_A_test", "ft_dev_B.txt", "ft_test_B.txt"]))

# Confusion Matrix
  dat_dev['Relevance_predicted_ft'] = dat_dev['Document'].apply(lambda x: model_ft_A.predict(x)[0][0])
  dat_dev['Polarity_predicted_ft'] = dat_dev['Document'].apply(lambda x: model_ft_B.predict(x)[0][0])
  dat_test['Relevance_predicted_ft'] = dat_test['Document'].apply(lambda x: model_ft_A.predict(x)[0][0])
  dat_test['Polarity_predicted_ft'] = dat_test['Document'].apply(lambda x: model_ft_B.predict(x)[0][0])
  confusion_matrix(dat_dev['Relevance'], dat_dev['Relevance_predicted_ft'], normalize= 'all')
  confusion_matrix(dat_test['Relevance'], dat_test['Relevance_predicted_ft'], normalize= 'all')
  confusion_matrix(dat_dev['Polarity'], dat_dev['Polarity_predicted_ft'], normalize= 'all')
  confusion_matrix(dat_test['Polarity'], dat_test['Polarity_predicted_ft'], normalize= 'all')