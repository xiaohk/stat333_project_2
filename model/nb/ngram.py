#!/usr/bin/python3
import pandas as pd
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from pickle import load, dump
from os.path import exists
import nb

N = 3 
TRAIN_TEXT = '../../data/text_train.csv'
TEST_TEXT = '../../data/text_test.csv'
TUNE_TEXT = '../../data/text_validate.csv'
MINI_TEXT = '../../data/text_mini.csv'
PREDICTION_CSV = './config/prediction.csv'
SAVE_DATA = './config/{}_gram_words.pickle'.format(N)
DISCARD = 50
NEXT = 3000 
RELOAD = False
STOP_WORDS = set(stopwords.words('english'))


def make_training_example(input_csv, all_words):
    """ Make training examples for the bayesian classifier from nltk."""
    table = pd.read_csv(input_csv)
    # Use string to represent the categories
    text, label = table['text'], list(map(str, table['stars']))

    # Each example is a dictionary with a label
    train_example = []
    length = len(text)
    for i in range(length):
        # Remove stopping words
        text_set = set(ngrams([w.lower() for w in word_tokenize(text[i])
                              if w not in STOP_WORDS], N))
        text_dict = {word: (word in text_set) for word in all_words}
        train_example.append((text_dict, label[i]))
        print("Encoding {}/{} example".format(i + 1, length), end='\r')
    print('\n')

    return train_example


def get_all_words(*input_csv):
    """ Get the vocabulary bank for all words used in the text."""
    text = []
    for csv in input_csv:
        text += pd.read_csv(csv)['text'].tolist()

    # We want to track the frequency of each word, so we then can select
    # features using the word frequency. We also want a dictionary to map words
    # to an integer, so we can save memory for not using string.
    curr_index = 0
    dictionary = {}

    words = []
    freq = []

    for review in text:
        for word in ngrams([w.lower() for w in word_tokenize(review)
                            if w not in STOP_WORDS], N):
            if word in dictionary:
                freq[dictionary[word]] += 1
            else:
                dictionary[word] = curr_index
                words.append(word)
                freq.append(1)
                curr_index += 1
    print(words)
    return dictionary, words, freq


def main():
    # Prepare for the training
    print("Start building the vocabulary bank.")

    if exists(SAVE_DATA) and not RELOAD:
        # Directly use saved data
        with open(SAVE_DATA, 'rb') as fp:
            dic, words, freq = load(fp)
    else:
        # Load and save data
        dic, words, freq = get_all_words(TRAIN_TEXT, TUNE_TEXT, TEST_TEXT)
        with open(SAVE_DATA, 'wb') as fp:
            data = [dic, words, freq]
            dump(data, fp)

    print(len(words))

    all_words = nb.select_words(words, freq, 0, NEXT)

    print("Use {} features".format(len(all_words)))

    print("Start encoding training examples.")
    train_examples = make_training_example(TRAIN_TEXT, all_words)

    # Train a Bayesian classifier
    classifier = NaiveBayesClassifier.train(train_examples)
    classifier.show_most_informative_features()

    # Write prediction output
    nb.predict(classifier, all_words, PREDICTION_CSV, TEST_TEXT, TUNE_TEXT)


if __name__ == '__main__':
    main()
