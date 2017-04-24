#!/usr/bin/python3
# import numpy as np
import pandas as pd
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from json import load, dump
from os.path import exists

TRAIN_TEXT = '../../data/text_train.csv'
TEST_TEXT = '../../data/text_test.csv'
TUNE_TEXT = '../../data/text_validate.csv'
MINI_TEXT = '../../data/text_mini.csv'
PREDICTION_CSV = './config/prediction.csv'
SAVE_DATA = './config/words.json'
DISCARD = 50
NEXT = 1000
RELOAD = False


def make_training_example(input_csv, all_words):
    """ Make training examples for the bayesian classifier from nltk."""
    table = pd.read_csv(input_csv)
    # Use string to represent the categories
    text, label = table['text'], list(map(str, table['stars']))

    # Each example is a dictionary with a label
    train_example = []
    length = len(text)
    for i in range(length):
        text_set = set(word_tokenize(text[i]))
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
        for word in word_tokenize(review):
            low = word.lower()
            if low in dictionary:
                freq[dictionary[low]] += 1
            else:
                dictionary[low] = curr_index
                words.append(low)
                freq.append(1)
                curr_index += 1

    return dictionary, words, freq


def select_words(words, freq, discard, use_next):
    """ It is unrealistic to choose all the words as our features, so we
        have use some method to select a subset of words.
        In this method, we discard top x words (function words), then use
        the next y words as features.
    """
    # Sort two lists systematically
    freq, words = zip(*sorted(zip(freq, words), reverse=True))
    print(words[:50])
    # Choose the next 1000 words after first 50 words
    # Its likely to discard 'great' and 'good', we add it back
    return set(words[discard: discard + use_next])


def predict(classifier, all_words, output_csv, *predict_csv):
    """ Use trained classifier to make prediction on the given csv file.
        write the prediction to output_csv.
    """
    prediction = []
    prediction_id = []

    # Support multiple predict_csv, and following the given order
    for csv in predict_csv:
        table = pd.read_csv(csv)
        length = len(table)
        text, ids = table['text'], table['Id']

        for i in range(len(text)):
            token = set(word_tokenize(text[i]))
            # Predict and record the result
            prediction.append(classifier.classify(
                {word: (word in token) for word in all_words}))
            prediction_id.append(ids[i])

            print("Predicting {}/{} of table {}".format(i, length,
                                                        csv), end='\r')
        print('\n')

    # Write to the output_csv following kaggle format
    with open(output_csv, 'w') as output:
        output.write('"Id","Prediction"\n')
        for i in range(len(prediction)):
            output.write("{},{}\n".format(prediction_id[i], prediction[i]))


def main():
    # Prepare for the training
    print("Start building the vocabulary bank.")

    if exists(SAVE_DATA) and not RELOAD:
        # Directly use saved data
        with open(SAVE_DATA, 'r') as fp:
            dic, words, freq = load(fp)
    else:
        # Load and save data
        dic, words, freq = get_all_words(TRAIN_TEXT, TUNE_TEXT, TEST_TEXT)
        with open(SAVE_DATA, 'w') as fp:
            data = [dic, words, freq]
            dump(data, fp, indent=4)

    print(len(words))

    all_words = select_words(words, freq, DISCARD, NEXT)

    print("Use {} features".format(len(all_words)))

    print("Start encoding training examples.")
    train_examples = make_training_example(TRAIN_TEXT, all_words)

    # Train a Bayesian classifier
    nb = NaiveBayesClassifier.train(train_examples)
    nb.show_most_informative_features()

    # Write prediction output
    predict(nb, all_words, PREDICTION_CSV, TEST_TEXT, TUNE_TEXT)


if __name__ == '__main__':
    main()
