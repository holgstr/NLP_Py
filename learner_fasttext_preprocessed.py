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

# Train naive fastText
model_ft_A = fasttext.train_supervised('ft_train_A.txt', wordNgrams = 2, epoch = 25)
model_ft_B = fasttext.train_supervised('ft_train_B.txt', wordNgrams = 2, epoch = 25)

# Train (manually) autotuned fastText
for x in range(100):
    s_wordNgrams = np.random.randint(1, 7)
    s_epoch = np.random.randint(10, 61)
    s_lr = np.random.uniform(0.1, 1.0)
    s_dim = np.random.randint(80, 181)
    s_ws = np.random.randint(3, 8)
    model_ft_A_tuned = fasttext.train_supervised('ft_train_A.txt',
                                           wordNgrams=s_wordNgrams,
                                           epoch=s_epoch,
                                           lr= s_lr,
                                           dim=s_dim,
                                           ws=s_ws)
    model_ft_B_tuned = fasttext.train_supervised('ft_train_B.txt',
                                           wordNgrams=s_wordNgrams,
                                           epoch=s_epoch,
                                           lr= s_lr,
                                           dim=s_dim,
                                           ws=s_ws)
    current_f1_A = m_f1(model_ft_A_tuned, 'ft_dev_A.txt')
    current_f1_B = m_f1(model_ft_B_tuned, 'ft_dev_B.txt')
    if x == 0:
        best_model_A = model_ft_A_tuned
        best_f1_A = current_f1_A
        best_model_B = model_ft_B_tuned
        best_f1_B = current_f1_B
    if current_f1_A > best_f1_A:
        best_f1_A = current_f1_A
        best_model_A = model_ft_A_tuned
    if current_f1_B > best_f1_B:
        best_f1_B = current_f1_B
        best_model_B = model_ft_B_tuned

# Test fastText
# Derive Micro-F1 manually to make sure I understand how it is computed
def m_f1(model, eval_file):
  p = model.test(eval_file)[1]
  r = model.test(eval_file)[2]
  return 2*(p*r/(p+r))

f1_ft = (pd.Series([m_f1(model_ft_A, 'ft_dev_A.txt'), m_f1(model_ft_A, 'ft_test_A.txt'),
                    m_f1(model_ft_B, 'ft_dev_B.txt'),m_f1(model_ft_B, 'ft_test_B.txt')],
                   index = ["ft_A_dev", "ft_A_test", "ft_dev_B.txt", "ft_test_B.txt"]))
f1_ft_tuned = (pd.Series([m_f1(best_model_A, 'ft_dev_A.txt'), m_f1(best_model_A, 'ft_test_A.txt'),
                    m_f1(best_model_B, 'ft_dev_B.txt'),m_f1(best_model_B, 'ft_test_B.txt')],
                   index = ["ft_A_dev", "ft_A_test", "ft_dev_B.txt", "ft_test_B.txt"]))