# Pakete importieren
import csv
import numpy as np
import pandas as pd
import sklearn.feature_selection
from gensim.utils import simple_preprocess
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
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
dat_syn = nb_df_preprocess(dat_syn)
dat_dia = nb_df_preprocess(dat_dia)

# Cross Validation vorbereiten, Folds für CV zuweisen
np.random.seed(2021)
folds = np.repeat([1, 2 ,3, 4, 5, 6, 7, 8, 9, 10], dat_train.count()[0]/10)
np.random.shuffle(folds)
dat_train['Fold'] = folds
vectorizer = CountVectorizer()

# Compute Micro-F1 for given model and eval_file
def m_f1(model, eval_file, task):
  if task == 'A':
    problem = 'Relevance'
  else:
    problem = 'Polarity'
  return f1_score(eval_file[problem], model.predict(vectorizer.transform(eval_file['Document'])), average='micro')

# Autotune Naive Bayes
def autotune_NB(type):
  for x in range(50):
    s_alpha = np.random.uniform(0.001, 3.0)
    current_f1_A_nb = []
    current_f1_B_nb = []
    for y in range(10):  # For each fold
        train = dat_train[dat_train['Fold'] != (y + 1)]
        valid = dat_train[dat_train['Fold'] == (y + 1)]
        if type == "Bernoulli":
            model_nb_A_tuned = BernoulliNB(alpha=s_alpha)
            model_nb_B_tuned = BernoulliNB(alpha=s_alpha)
        if type == "Complement":
            model_nb_A_tuned = ComplementNB(alpha=s_alpha)
            model_nb_B_tuned = ComplementNB(alpha=s_alpha)
        else:
            model_nb_A_tuned = MultinomialNB(alpha=s_alpha)
            model_nb_B_tuned = MultinomialNB(alpha=s_alpha)
        model_nb_A_tuned.fit(vectorizer.fit_transform(train['Document']), np.array(train['Relevance']))
        model_nb_B_tuned.fit(vectorizer.fit_transform(train['Document']), np.array(train['Polarity']))
        current_f1_A_nb.append(m_f1(model_nb_A_tuned, valid, 'A'))
        current_f1_B_nb.append(m_f1(model_nb_B_tuned, valid, 'B'))
    current_f1_A_nb = np.mean(current_f1_A_nb)
    current_f1_B_nb = np.mean(current_f1_B_nb)
    if x == 0:
        best_alpha_nb_A = s_alpha
        best_f1_A_nb = current_f1_A_nb
        best_alpha_nb_B = s_alpha
        best_f1_B_nb = current_f1_B_nb
    if current_f1_A_nb > best_f1_A_nb:
        best_f1_A_nb = current_f1_A_nb
        best_alpha_nb_A = s_alpha
    if current_f1_B_nb > best_f1_B_nb:
        best_f1_B_nb = current_f1_B_nb
        best_alpha_nb_B = s_alpha
    print("Durchlauf", x + 1, "/50")
  return [best_alpha_nb_A, best_alpha_nb_B, best_f1_A_nb, best_f1_B_nb, type]

best_alpha_mnb = autotune_NB('Multinomial')
best_alpha_bnb = autotune_NB('Bernoulli')
best_alpha_cnb = autotune_NB('Complement')

# Train Naive Bayes with optimal hyperparameters
model_mnb_A_tuned = MultinomialNB(alpha=best_alpha_mnb[0])
model_mnb_B_tuned = MultinomialNB(alpha=best_alpha_mnb[1])
model_bnb_A_tuned = BernoulliNB(alpha=best_alpha_bnb[0])
model_bnb_B_tuned = BernoulliNB(alpha=best_alpha_bnb[1])
model_cnb_A_tuned = ComplementNB(alpha=best_alpha_cnb[0])
model_cnb_B_tuned = ComplementNB(alpha=best_alpha_cnb[1])
model_mnb_A_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
model_mnb_B_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))
model_bnb_A_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
model_bnb_B_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))
model_cnb_A_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Relevance']))
model_cnb_B_tuned.fit(vectorizer.fit_transform(dat_train['Document']), np.array(dat_train['Polarity']))

# Test Naive Bayes
f1_nb = pd.DataFrame(data={'data': ["syn_A", "dia_A", "syn_B", "dia_B"],
                           'tuned MNB': [m_f1(model_mnb_A_tuned, dat_syn, 'A'),
                                         m_f1(model_mnb_A_tuned, dat_dia, 'A'),
                                         m_f1(model_mnb_B_tuned, dat_syn, 'B'),
                                         m_f1(model_mnb_B_tuned, dat_dia, 'B')],
                           'tuned BNB': [m_f1(model_bnb_A_tuned, dat_syn, 'A'),
                                         m_f1(model_bnb_A_tuned, dat_dia, 'A'),
                                         m_f1(model_bnb_B_tuned, dat_syn, 'B'),
                                         m_f1(model_bnb_B_tuned, dat_dia, 'B')],
                           'tuned CNB': [m_f1(model_cnb_A_tuned, dat_syn, 'A'),
                                         m_f1(model_cnb_A_tuned, dat_dia, 'A'),
                                         m_f1(model_cnb_B_tuned, dat_syn, 'B'),
                                         m_f1(model_cnb_B_tuned, dat_dia, 'B')]})
f1_nb



### Text Example: Select 20 words by highest Chi2 Criterion
vectorizer = CountVectorizer(max_df=1000)
train = vectorizer.fit_transform(dat_train['Document'])
train = train.toarray()
listchi2 = chi2(train, np.array(dat_train['Polarity']))[0]
ind = np.argpartition(listchi2, -20)[-20:]
print(np.array(vectorizer.get_feature_names())[ind])
