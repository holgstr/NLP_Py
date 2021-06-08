# Pakete importieren
import csv, re, os
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
  # Introduce Tokens for Twitter Usernames, but keep official DB Twitter Accounts as such
  df['Document'] = df['Document'].str.replace("@DB_Bahn", '<ACC_DB_Bahn>', regex = True)
  df['Document'] = df['Document'].str.replace("@Bahn_Info", '<ACC_Bahn_Info>', regex=True)
  df['Document'] = df['Document'].str.replace("@(\s?)(\w{1,15})", '<TwitterUserName>', regex=True)
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
