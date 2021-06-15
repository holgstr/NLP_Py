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

# Cross Validation vorbereiten, Folds für CV zuweisen
folds = np.repeat([1, 2 ,3, 4], dat_train.count()[0]/4)
np.random.shuffle(folds)
dat_train['Fold'] = folds

# Compute Micro-F1 for given model and eval_file
def m_f1(model, eval_file, task):
  if task == 'A':
    problem = 'Relevance'
  else:
    problem = 'Polarity'
  return f1_score(eval_file[problem], model.predict(vectorizer.transform(eval_file['Document'])), average='micro')

# Autotune Naive Bayes
for x in range(50):
    s_alpha = (3/50*(x+1))
    current_f1_A_mnb = []
    current_f1_B_mnb = []
    current_f1_A_bnb = []
    current_f1_B_bnb = []
    for y in range(4):  # For each fold
        train = dat_train[dat_train['Fold'] != (y + 1)]
        valid = dat_train[dat_train['Fold'] == (y + 1)]
        model_mnb_A_tuned = MultinomialNB(alpha=s_alpha)
        model_mnb_B_tuned = MultinomialNB(alpha=s_alpha)
        model_bnb_A_tuned = BernoulliNB(alpha=s_alpha)
        model_bnb_B_tuned = BernoulliNB(alpha=s_alpha)
        model_mnb_A_tuned.fit(vectorizer.fit_transform(train['Document']), np.array(train['Relevance']))
        model_mnb_B_tuned.fit(vectorizer.fit_transform(train['Document']), np.array(train['Polarity']))
        model_bnb_A_tuned.fit(vectorizer.fit_transform(train['Document']), np.array(train['Relevance']))
        model_bnb_B_tuned.fit(vectorizer.fit_transform(train['Document']), np.array(train['Polarity']))
        current_f1_A_mnb.append(m_f1(model_mnb_A_tuned, valid, 'A'))
        current_f1_B_mnb.append(m_f1(model_mnb_B_tuned, valid, 'B'))
        current_f1_A_bnb.append(m_f1(model_bnb_A_tuned, valid, 'A'))
        current_f1_B_bnb.append(m_f1(model_bnb_B_tuned, valid, 'B'))
    current_f1_A_mnb = np.mean(current_f1_A_mnb)
    current_f1_B_mnb = np.mean(current_f1_B_mnb)
    current_f1_A_bnb = np.mean(current_f1_A_bnb)
    current_f1_B_bnb = np.mean(current_f1_B_bnb)
    if x == 0:
        best_alpha_mnb_A = s_alpha
        best_f1_A_mnb = current_f1_A_mnb
        best_alpha_mnb_B = s_alpha
        best_f1_B_mnb = current_f1_B_mnb
        best_alpha_bnb_A = s_alpha
        best_f1_A_bnb = current_f1_A_bnb
        best_alpha_bnb_B = s_alpha
        best_f1_B_bnb = current_f1_B_bnb
    if current_f1_A_mnb > best_f1_A_mnb:
        best_f1_A_mnb = current_f1_A_mnb
        best_alpha_mnb_A = s_alpha
    if current_f1_B_mnb > best_f1_B_mnb:
        best_f1_B_mnb = current_f1_B_mnb
        best_alpha_mnb_B = s_alpha
    if current_f1_A_bnb > best_f1_A_bnb:
        best_f1_A_bnb = current_f1_A_bnb
        best_alpha_bnb_A = s_alpha
    if current_f1_B_bnb > best_f1_B_bnb:
        best_f1_B_bnb = current_f1_B_bnb
        best_alpha_bnb_B = s_alpha
    print("Durchlauf", x+1, "/50")

# Train Naive Bayes with optimal hyperparameters
model_mnb_A_tuned = MultinomialNB(alpha=best_alpha_mnb_A)
model_mnb_B_tuned = MultinomialNB(alpha=best_alpha_mnb_B)
model_bnb_A_tuned = BernoulliNB(alpha=best_alpha_bnb_A)
model_bnb_B_tuned = BernoulliNB(alpha=best_alpha_bnb_B)
model_mnb_A_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
model_mnb_B_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))
model_bnb_A_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
model_bnb_B_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))

# Test Naive Bayes
f1_nb = pd.DataFrame(data={'data': ["dev_A", "syn_A", "dia_A", "dev_B", "syn_B", "dia_B"],
                           'tuned MNB': [m_f1(model_mnb_A_tuned, dat_dev, 'A'), m_f1(model_mnb_A_tuned, dat_syn, 'A'),
                                         m_f1(model_mnb_A_tuned, dat_dia, 'A'), m_f1(model_mnb_B_tuned, dat_dev, 'B'),
                                         m_f1(model_mnb_B_tuned, dat_syn, 'B'), m_f1(model_mnb_B_tuned, dat_dia, 'B')],
                           'tuned BNB': [m_f1(model_bnb_A_tuned, dat_dev, 'A'), m_f1(model_bnb_A_tuned, dat_syn, 'A'),
                                         m_f1(model_bnb_A_tuned, dat_dia, 'A'), m_f1(model_bnb_B_tuned, dat_dev, 'B'),
                                         m_f1(model_bnb_B_tuned, dat_syn, 'B'), m_f1(model_bnb_B_tuned, dat_dia, 'B')]})
f1_nb