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

# Duplikate aus den Trainingsdaten löschen, Daten zusammenführen
dat_train = dat_train.drop_duplicates(subset=['Document'])
dat_dev = dat_dev.drop_duplicates(subset=['Document'])
dat_train = dat_train.append(dat_dev, ignore_index=True)
dat_train2 = dat_train

# Daten vorbereiten für fasttext
def fasttext_df_preprocess(df, df_name):
  # Delete NAs
  df = df.dropna(subset=['URL', 'Document']).reset_index(drop=True)
  # Adding Domain Names to Document
  domain = df['URL'].str.extract(
      pat='((\w|-)*(?=.com/|.de/|.at/|.eu/|.ch/|.net/|.info/|.org/|.me/|.tv/|.travel/))').iloc[:, 0]
  df['Document'] = domain.str.cat(df['Document'], sep=" ", na_rep=" ")
  # Tokenization
  df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
  # Preparing Labels for fastText
  df['Relevance'] = df['Relevance'].astype(str).apply(lambda x: '__label__' + x)
  df['Polarity'] = df['Polarity'].astype(str).apply(lambda x: '__label__' + x)
  # Save Files for fastText
  df[['Document', 'Relevance']].to_csv(os.path.join('ft_' + df_name + '_A.txt'), index=False, sep=' ', header=None, quoting=csv.QUOTE_NONE,
                                             quotechar="", escapechar=" ")
  df[['Document', 'Polarity']].to_csv(os.path.join('ft_' + df_name + '_B.txt'), index=False, sep=' ', header=None,
                                             quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
  return df

dat_train = fasttext_df_preprocess(dat_train, 'train')
dat_syn = fasttext_df_preprocess(dat_syn, 'syn')
dat_dia = fasttext_df_preprocess(dat_dia, 'dia')

# Cross Validation vorbereiten, Folds für CV zuweisen
np.random.seed(2021)
folds = np.repeat([1, 2 ,3, 4, 5], dat_train.count()[0]/5)
np.random.shuffle(folds)
dat_train['Fold'] = folds

# Manually conduct hyperparameter optimization using CV on train+dev, as fastText-native autotuning search space is too large
f1_time_A = []
f1_time_B = []
for x in range(100): # For each hyperparameter configuration
    params = [np.random.randint(1, 5), np.random.randint(5, 61), np.random.uniform(0.1, 1.0), np.random.randint(3, 8)]
    current_f1_A = []
    current_f1_B = []
    for y in range(10): # For each fold
        train = dat_train[dat_train['Fold'] != (y+1)]
        valid = dat_train[dat_train['Fold'] == (y+1)]
        train[['Document', 'Relevance']].to_csv('train_A.txt', index=False, sep=' ', header=None,
                                            quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
        train[['Document', 'Polarity']].to_csv('train_B.txt', index=False, sep=' ', header=None,
                                                  quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
        valid[['Document', 'Relevance']].to_csv('valid_A.txt', index=False, sep=' ', header=None,
                                                  quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
        valid[['Document', 'Polarity']].to_csv('valid_B.txt', index=False, sep=' ', header=None,
                                                  quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
        model_ft_A_tuned = fasttext.train_supervised('train_A.txt',
                                           wordNgrams = params[0],
                                           epoch = params[1],
                                           lr = params[2],
                                           ws = params[3])
        model_ft_B_tuned = fasttext.train_supervised('train_B.txt',
                                           wordNgrams = params[0],
                                           epoch =params[1],
                                           lr = params[2],
                                           ws = params[3])
        current_f1_A.append(model_ft_A_tuned.test('valid_A.txt')[1])
        current_f1_B.append(model_ft_B_tuned.test('valid_B.txt')[1])
    current_f1_A = np.mean(current_f1_A)
    current_f1_B = np.mean(current_f1_B)
    if x == 0:
        best_params_A = params
        best_f1_A = current_f1_A
        best_params_B = params
        best_f1_B = current_f1_B
    if current_f1_A > best_f1_A:
        best_f1_A = current_f1_A
        best_params_A = params
    if current_f1_B > best_f1_B:
        best_f1_B = current_f1_B
        best_params_B = params
    f1_time_A.append(best_f1_A)
    f1_time_B.append(best_f1_B)
    print("Durchlauf", x+1, "/100")

# Train fastText with optimal hyperparameters
best_model_ft_A = fasttext.train_supervised('ft_train_A.txt',
                                           wordNgrams = best_params_A[0],
                                           epoch = best_params_A[1],
                                           lr = best_params_A[2],
                                           ws = best_params_A[3])
best_model_ft_B = fasttext.train_supervised('ft_train_B.txt',
                                           wordNgrams = best_params_B[0],
                                           epoch = best_params_B[1],
                                           lr = best_params_B[2],
                                           ws = best_params_B[3])

# Train fastText with naive specs
model_ft_A = fasttext.train_supervised('ft_train_A.txt')
model_ft_B = fasttext.train_supervised('ft_train_B.txt')

# Test fastText
f1_ft = pd.DataFrame(data={'data': ["syn_A", "dia_A", "syn_B", "dia_B"],
                           'naive': [model_ft_A.test('ft_syn_A.txt')[1],
                                     model_ft_A.test('ft_dia_A.txt')[1],
                                     model_ft_B.test('ft_syn_B.txt')[1],
                                     model_ft_B.test('ft_dia_B.txt')[1]],
                           'tuned': [best_model_ft_A.test('ft_syn_A.txt')[1],
                                     best_model_ft_A.test('ft_dia_A.txt')[1],
                                     best_model_ft_B.test('ft_syn_B.txt')[1],
                                     best_model_ft_B.test('ft_dia_B.txt')[1]]})
f1_ft

# Zusatzbeispiel: Generate Word Embeddings on Data
dat_train2 = dat_train2.dropna(subset=['URL', 'Document']).reset_index(drop=True)
dat_train2.iloc[:, 1] = dat_train2.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
dat_train2[['Document']].to_csv('ft_corpus.txt', index=False, sep=' ', header=None,
                                     quoting=csv.QUOTE_NONE,
                                     quotechar="", escapechar=" ")
model1 = fasttext.train_unsupervised('ft_corpus.txt', minCount = 5, minn = 2, maxn = 5, dim = 300)
model1.get_nearest_neighbors('gleis')