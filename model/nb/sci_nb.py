#!/usr/bin/python3
# import numpy as np
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from pickle import load, dump
from os.path import exists

TRAIN_TEXT = '../../data/text_train.csv'
TEST_TEXT = '../../data/text_test.csv'
VALI_TEXT = '../../data/text_validate.csv'
MINI_TEXT = '../../data/text_mini.csv'
PREDICTION_CSV = '../../static/prediction.csv'
VECTOR = '../../static/tf_vector_2_gram.pickle'
MATRIX = '../../static/tf_matrix_2_gram.pickle'

STOP = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")
LOAD_NEW = False

# Tunable parameters
MAX_DF = 1.0  # We already have stop words, probably don't need this
MIN_DF = 2  # Discard words which not show up twice in all document
MAX_FEATURE = None  # IF no memory, tune this down
MIN_N = 1
MAX_N = 1  # Uni-gram


def tokenize(text):
    """ Helper function for TFIDF vector from sklearn. We use nltk's tokenize
        function here. Also use snowball as stemmer (the middle agressive
        one).
    """
    # Filter out stop words, tokenize the text
    useful_token = [w.lower() for w in word_tokenize(text) if w not in STOP]

    # Stemming the tokens
    stemmed_token = [STEMMER.stem(t) for t in useful_token]

    return stemmed_token


def vectorize_text(train_txt, vali_txt, test_txt, vector, matrix, re_load,
                   min_df=1, max_df=1.0, max_feature=None, min_n=1, max_n=1):
    """ Feature engineering from the raw text input. """
    # If there is saved model, then just use it
    if exists(vector) and exists(matrix) and not re_load:
        # Get train length
        table_train = pd.read_csv(train_txt)

        # Load stored data
        all_mat = load(open(MATRIX, 'rb'))
        x_train = all_mat[:table_train.shape[0]]
        tf = load(open(VECTOR, 'rb'))

    else:
        # Read all files
        table_train = pd.read_csv(train_txt)
        table_test = pd.read_csv(vali_txt)
        table_vali = pd.read_csv(test_txt)

        text_train = table_train['text'].tolist()
        text_test = table_test['text'].tolist()
        text_vali = table_vali['text'].tolist()

        # We want to have a overall vocabulary bank as `np.py`, so we combine
        # all the text first
        all_text = text_train + text_test + text_vali

        # Record the length so we can recover the training set
        train_length = len(text_train)

        # Initialize TFID arguments
        # Only work for English, discard all Chinese
        tf = TfidfVectorizer(min_df=min_df, max_features=max_feature,
                             strip_accents='ascii', analyzer='word',
                             tokenizer=tokenize, ngram_range=(min_n, max_n))

        # Vectorize all, and transform (more efficient than fit + transform)
        all_mat = tf.fit_transform(all_text)

        # Recover the training data
        x_train = all_mat[:train_length]

        # Store the fitted matrix and tf_vectorizor
        dump(all_mat, open(matrix, 'wb'))
        dump(tf, open(vector, 'wb'))

    print("Successfully load TF-IDF matrix, with shape {}.".format(
        x_train.shape))

    return tf, all_mat, x_train


def train_nb(x_train, y_train):
    """ Train a multinomial nb model."""
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    print("10-CV average accuracy: {}".format(
          np.mean(cross_val_score(nb, x_train, y_train, cv=10,
                                  scoring=score))))

    return nb


def score(estimator, x_test, y_test):
    """ Use mean squared error as score for cv."""
    probs = estimator.predict_proba(x_test)
    result = np.zeros(x_test.shape[0])
    for i in range(probs.shape[0]):
        result[i] = dis_to_conti(probs[i])
    y_int = np.array(list(map(int, y_test)))
    score = np.sqrt(np.mean(np.square(result - y_int)))
    return score


def dis_to_conti(probability):
    """ The kaggle grading is unfair, so I want to force bayesian classifier
        gives a continuous result.
    """
    return sum(probability * np.arange(1, 6))


def main():
    print("Start vectorizing...")
    tf, all_mat, x_train = vectorize_text(TRAIN_TEXT, TEST_TEXT, VALI_TEXT,
                                          VECTOR, MATRIX, LOAD_NEW,
                                          min_df=MIN_DF, max_df=MAX_DF,
                                          max_feature=MAX_FEATURE,
                                          min_n=MIN_N, max_n=MAX_N)

    # Make label for train_v
    table_train = pd.read_csv(TRAIN_TEXT)
    # Use string to represent the categories
    y_train = list(map(str, table_train['stars']))

    print("Start training...")
    nb = train_nb(x_train, y_train)


if __name__ == '__main__':
    main()
