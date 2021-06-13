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

# Compute Micro-F1 for given model and eval_file
def m_f1(model, eval_file, task):
  if task == 'A':
    problem = 'Relevance'
  else:
    problem = 'Polarity'
  return f1_score(eval_file[problem], model.predict(vectorizer.transform(eval_file['Document'])), average='micro')

# Train Naive Bayes Classifier
vectorizer = CountVectorizer()
model_nb_A = MultinomialNB(alpha=1.0)
model_nb_A.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
model_nb_B = MultinomialNB(alpha=1.0)
model_nb_B.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))

# Autotune Naive Bayes
for x in range(250):
    s_alpha = np.random.uniform(0.01, 1.0)
    model_nb_A_tuned = MultinomialNB(alpha=s_alpha)
    model_nb_B_tuned = MultinomialNB(alpha=s_alpha)
    model_nb_A_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
    model_nb_B_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))
    current_f1_A = m_f1(model_nb_A_tuned, dat_dev, 'A')
    current_f1_B = m_f1(model_nb_B_tuned, dat_dev, 'B')
    if x == 0:
        best_model_nb_A = model_nb_A_tuned
        best_f1_A = current_f1_A
        best_model_nb_B = model_nb_B_tuned
        best_f1_B = current_f1_B
    if current_f1_A > best_f1_A:
        best_f1_A = current_f1_A
        best_model_nb_A = model_nb_A_tuned
    if current_f1_B > best_f1_B:
        best_f1_B = current_f1_B
        best_model_nb_B = model_nb_B_tuned
    print("Durchlauf", x+1, "/250")

# Test Naive Bayes
f1_nb = pd.DataFrame(data={'data': ["dev_A", "syn_A", "dia_A", "dev_B", "syn_B", "dia_B"],
                           'naive': [m_f1(model_nb_A, dat_dev, 'A'), m_f1(model_nb_A, dat_syn, 'A'),
                                     m_f1(model_nb_A, dat_dia, 'A'), m_f1(model_nb_B, dat_dev, 'B'),
                                     m_f1(model_nb_B, dat_syn, 'B'), m_f1(model_nb_B, dat_dia, 'B')],
                           'tuned': [m_f1(best_model_nb_A, dat_dev, 'A'), m_f1(best_model_nb_A, dat_syn, 'A'),
                                     m_f1(best_model_nb_A, dat_dia, 'A'), m_f1(best_model_nb_B, dat_dev, 'B'),
                                     m_f1(best_model_nb_B, dat_syn, 'B'), m_f1(best_model_nb_B, dat_dia, 'B')]})
f1_nb