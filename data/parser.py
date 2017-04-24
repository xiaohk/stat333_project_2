import pandas as pd
import numpy as np
import re
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))

TRAIN_CSV = './Yelp_train.csv'
TRAIN_OUT = './text_train.csv'
TRAIN_TXT = './train.txt'
MINI_CSV = './Yelp_mini.csv'
MINI_OUT = './text_mini.csv'
TUNE_CSV = './Yelp_validate.csv'
TUNE_OUT = './text_validate.csv'
TEST_CSV = './Yelp_test.csv'
TEST_OUT = './text_test.csv'
TRAIN_NEW = './new_train.csv'
TEST_NEW = './new_test.csv'
TUNE_NEW = './new_validate.csv'
MINI_NEW = './new_mini.csv'


def make_text_label(input_csv, output_csv):
    """Convert given csv file to a text and label pair csv file."""
    # Using pandas to convert tables easily
    table = pd.read_csv(input_csv)
    new_table = table[['stars', 'text']].copy()

    # Add a cleaned text column
    new_table['clean'] = pd.Series([clean_text(t) for t in new_table['text']])

    # Write to a new file
    new_table.to_csv(output_csv, sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)


def make_text(input_csv, output_csv):
    """ Convert given csv file to a raw text csv file."""
    # Using pandas to convert tables easily
    table = pd.read_csv(input_csv)
    new_table = table[['text', 'Id']].copy()

    # Add a cleaned text column
    new_table['clean'] = pd.Series([clean_text(t) for t in new_table['text']])

    # Write to a new file
    new_table.to_csv(output_csv, sep=',', encoding='utf-8', header=True,
                     doublequote=True, index=False)


def clean_text(text):
    """ Cleaning the format for the review text."""
    # Make punctuations slitted from the front words
    new_text = re.sub(r"""([.,;!?()$"\\/#%@&*\{\}\[\]:~`])""", r' \1 ', text)

    # For `'`, we want it to concatenate with the char after it
    new_text = re.sub('\'', ' \'', new_text)

    # Combine 2+ spaces into one space
    new_text = re.sub('\s{2,}', ' ', new_text)

    # Filter out the new line sign
    new_text = re.sub('\n', ' ', new_text)

    # Concert all letters to lower case
    return new_text.lower()


def make_tagging(input_csv, output_csv):
    """ Add a tagging column for the input csv."""
    table = pd.read_csv(input_csv)
    row_num = table.shape[0]
    new_info = {'ADJ': np.zeros(row_num),
                'ADP': np.zeros(row_num),
                'ADV': np.zeros(row_num),
                'CONJ': np.zeros(row_num),
                'DET': np.zeros(row_num),
                'NOUN': np.zeros(row_num),
                'NUM': np.zeros(row_num),
                'PRT': np.zeros(row_num),
                'PRON': np.zeros(row_num),
                'VERB': np.zeros(row_num),
                '.': np.zeros(row_num),
                'X': np.zeros(row_num),
                'STOP': np.zeros(row_num)}

    for i in range(table.shape[0]):
        raw_text = table['text'][i]
        token = word_tokenize(raw_text)
        # Only use universal tagset
        tagged = pos_tag(token, tagset='universal')

        # Count the number of each tag for this text
        couter = FreqDist(t for w, t in tagged)

        # Count stop words
        num_stop = 0
        for t in token:
            num_stop += 1 if t in STOP else 0

        # Update to new_info
        for t, f in couter.items():
            new_info[t][i] = int(f)
        new_info['STOP'][i] = num_stop

    # Rename some odd names
    new_info['PUNC'] = new_info.pop('.')
    new_info['OTHER'] = new_info.pop('X')

    # Append those columns
    for k in new_info:
        table[k] = new_info[k]

    # Write to file
    table.to_csv(output_csv, sep=',', encoding='utf-8', header=True,
                 doublequote=True, index=False)


def main():
    # make_text_label(MINI_CSV, MINI_OUT)
    # make_text_label(TRAIN_CSV, TRAIN_OUT)
    # make_text(TUNE_CSV, TUNE_OUT)
    # make_text(TEST_CSV, TEST_OUT)
    make_tagging(TRAIN_CSV, TRAIN_NEW)
    make_tagging(TEST_CSV, TEST_NEW)
    make_tagging(TUNE_CSV, TUNE_NEW)


if __name__ == '__main__':
    main()
