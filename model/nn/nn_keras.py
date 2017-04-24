from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
import pandas as pd
import numpy as np

FEATURE_NUM = 34
OUTPUT_NUM = 5  # Have 5 classifications
BATCH_SIZE = 3
HIDDEN_LAYER_SIZE = 10
EPOCH = 10
TRAIN_CSV = '../../data/Yelp_train.csv'
TUNE_CSV = '../../data/Yelp_validate.csv'
TEST_CSV = '../../data/Yelp_test.csv'
MINI_CSV = '../../data/Yelp_mini.csv'


def encode_data(input_csv):
    """
    Make training features and labels.
    """
    table = pd.read_csv(input_csv, header=0).values
    # Feature and label list using native python list
    native_feature, native_label = [], []

    for row in table:
        if np.isnan(row[13]):
            # Sentiment score might be nah, we don't use those entires
            continue
        native_feature.append(np.hstack((row[4:7], row[11:]))
                              .astype(np.float))
        native_label.append(row[0])

    features = np.array(native_feature)
    # Use one-hot encoding for the label, to_categorical would use 0-index so
    # we subtract 1 here
    labels = to_categorical(np.array(native_label) - 1)

    return features, labels


def main():
    """Train a one hidden layer neural network."""

    # Get training and tuning set
    x_train, y_train = encode_data(TRAIN_CSV)
    x_tune, y_tune = encode_data(TUNE_CSV)

    # We use normal sequential neural network
    model = Sequential()
    # Input -> hidden
    model.add(Dense(HIDDEN_LAYER_SIZE, activation='relu',
                    input_dim=FEATURE_NUM))
    # Hidden -> output
    model.add(Dense(OUTPUT_NUM, activation='sigmoid'))
    # Configure learning parameters
    sgd = SGD(lr=0.001)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Start training
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
