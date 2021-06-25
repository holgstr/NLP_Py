# Pakete importieren
import csv
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import f1_score

# Daten einlesen
header_list = ["URL", "Document", "Relevance", "Polarity"]
dat_train = pd.read_csv("/Users/holgerlowe/Documents/NLP_Data/train.tsv", names = header_list, usecols = range(0,4), sep = '\t')
dat_dev = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/dev.tsv', names = header_list, usecols = range(0,4), sep = '\t')
dat_syn = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/syn.tsv', names = header_list, usecols = range(0,4), sep = '\t')
dat_dia = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/dia.tsv', names = header_list, usecols = range(0,4), sep = '\t')

# Duplikate aus den Trainingsdaten löschen, Daten zusammenführen
dat_train = dat_train.drop_duplicates(subset=['Document'])
dat_dev = dat_dev.drop_duplicates(subset=['Document'])
dat_train = dat_train.append(dat_dev, ignore_index=True)

# Daten vorbereiten für Naive Bayes
def nb_df_preprocess(df):
  # Delete NAs
  df = df.dropna(subset=['URL', 'Document']).reset_index(drop=True)
  # Adding Domain Names to Document
  domain = df['URL'].str.extract(
      pat='((\w|-)*(?=.com/|.de/|.at/|.eu/|.ch/|.net/|.info/|.org/|.me/|.tv/|.travel/))').iloc[:, 0]
  df['Document'] = domain.str.cat(df['Document'], sep=" ", na_rep=" ")
  # Tokenization
  df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
  return df
dat_train = nb_df_preprocess(dat_train)
dat_dev = nb_df_preprocess(dat_dev)
dat_syn = nb_df_preprocess(dat_syn)
dat_dia = nb_df_preprocess(dat_dia)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
vectorizer = CountVectorizer()
le = LabelEncoder()
lr = LogisticRegression()
train_y = le.fit_transform(dat_train['Relevance'])
train = vectorizer.fit_transform(dat_train['Document'])
lr.fit(train, train_y)
train['Relevance']