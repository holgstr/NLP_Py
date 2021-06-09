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
dat_syn = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/syn.tsv', names = header_list, usecols = range(0,4), sep = '\t')
dat_dia = pd.read_csv('/Users/holgerlowe/Documents/NLP_Data/dia.tsv', names = header_list, usecols = range(0,4), sep = '\t')

# Daten vorbereiten für fasttext
def fasttext_df_preprocess(df, df_name):
  # Delete NAs
  df = df.dropna(subset=['URL', 'Document']).reset_index(drop=True)
  # Adding Domain Names to Document
  domain = df['URL'].str.extract(
      pat='((\w|-)*(?=.com/|.de/|.at/|.eu/|.ch/|.net/|.info/|.org/|.me/|.tv/|.travel/))').iloc[:, 0]
  domain.str.cat(df['Document'], sep=" ", na_rep=" ")
  df['Document'] = domain.str.cat(df['Document'], sep=" ", na_rep=" ")
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
dat_syn = fasttext_df_preprocess(dat_syn, 'syn')
dat_dia = fasttext_df_preprocess(dat_dia, 'dia')

# Derive Micro-F1 manually to make sure I understand how it is computed
def m_f1(model, eval_file):
  p = model.test(eval_file)[1]
  r = model.test(eval_file)[2]
  return 2*(p*r/(p+r))

# Train fastText with naive specs
model_ft_A = fasttext.train_supervised('ft_train_A.txt', wordNgrams = 2, epoch = 25)
model_ft_B = fasttext.train_supervised('ft_train_B.txt', wordNgrams = 2, epoch = 25)

# Train (manually) autotuned fastText, as native autotuning disadvantageous properties
for x in range(250):
    s_wordNgrams = np.random.randint(1, 5)
    s_epoch = np.random.randint(5, 61)
    s_lr = np.random.uniform(0.1, 1.0)
    s_ws = np.random.randint(3, 8)
    model_ft_A_tuned = fasttext.train_supervised('ft_train_A.txt',
                                           wordNgrams=s_wordNgrams,
                                           epoch=s_epoch,
                                           lr= s_lr,
                                           ws=s_ws)
    model_ft_B_tuned = fasttext.train_supervised('ft_train_B.txt',
                                           wordNgrams=s_wordNgrams,
                                           epoch=s_epoch,
                                           lr= s_lr,
                                           ws=s_ws)
    current_f1_A = m_f1(model_ft_A_tuned, 'ft_dev_A.txt')
    current_f1_B = m_f1(model_ft_B_tuned, 'ft_dev_B.txt')
    if x == 0:
        best_model_ft_A_un = model_ft_A_tuned
        best_f1_A = current_f1_A
        best_model_ft_un = model_ft_B_tuned
        best_f1_B = current_f1_B
    if current_f1_A > best_f1_A:
        best_f1_A = current_f1_A
        best_model_ft_A_un = model_ft_A_tuned
    if current_f1_B > best_f1_B:
        best_f1_B = current_f1_B
        best_model_ft_B_un = model_ft_B_tuned
    print("Durchlauf", x+1, "/250")
del model_ft_A_tuned, model_ft_B_tuned
best_model_ft_A_un.save_model('best_model_ft_A_un.bin')
best_model_ft_B_un.save_model('best_model_ft_B_un.bin')

# Test fastText
f1_ft_unprocessed = pd.DataFrame(data={'data': ["dev_A", "syn_A", "dia_A", "dev_B", "syn_B", "dia_B"],
                           'naive': [m_f1(model_ft_A, 'ft_dev_A.txt'), m_f1(model_ft_A, 'ft_syn_A.txt'),
                                     m_f1(model_ft_A, 'ft_dia_A.txt'), m_f1(model_ft_B, 'ft_dev_B.txt'),
                                     m_f1(model_ft_B, 'ft_syn_B.txt'), m_f1(model_ft_B, 'ft_dia_B.txt')],
                           'tuned': [m_f1(best_model_ft_A_un, 'ft_dev_A.txt'), m_f1(best_model_ft_A_un, 'ft_syn_A.txt'),
                                     m_f1(best_model_ft_A_un, 'ft_dia_A.txt'), m_f1(best_model_ft_B_un, 'ft_dev_B.txt'),
                                     m_f1(best_model_ft_B_un, 'ft_syn_B.txt'), m_f1(best_model_ft_B_un, 'ft_dia_B.txt')]})
f1_ft_unprocessed
# Confusion Matrix
  dat_dev['Relevance_predicted_ft'] = dat_dev['Document'].apply(lambda x: model_ft_A.predict(x)[0][0])
  dat_dev['Polarity_predicted_ft'] = dat_dev['Document'].apply(lambda x: model_ft_B.predict(x)[0][0])
  dat_test['Relevance_predicted_ft'] = dat_test['Document'].apply(lambda x: model_ft_A.predict(x)[0][0])
  dat_test['Polarity_predicted_ft'] = dat_test['Document'].apply(lambda x: model_ft_B.predict(x)[0][0])
  confusion_matrix(dat_dev['Relevance'], dat_dev['Relevance_predicted_ft'], normalize= 'all')
  confusion_matrix(dat_test['Relevance'], dat_test['Relevance_predicted_ft'], normalize= 'all')
  confusion_matrix(dat_dev['Polarity'], dat_dev['Polarity_predicted_ft'], normalize= 'all')
  confusion_matrix(dat_test['Polarity'], dat_test['Polarity_predicted_ft'], normalize= 'all')


