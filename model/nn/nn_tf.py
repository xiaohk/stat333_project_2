import tensorflow as tf
import numpy as np
import pandas as pd

FEATURE_NUM = 10
OUTPUT_NUM = 5  # Have 5 classifications
BATCH_SIZE = 3
EPOCH = 100
TRAIN_CSV = '../../data/Yelp_train.csv'
TUNE_CSV = '../../data/Yelp_validate.csv'
TEST_CSV = '../../data/Yelp_test.csv'
MINI_CSV = '../../data/Yelp_mini.csv'
TRAIN_TFR = '../../data/Yelp_train.tfrecords'
MINI_TFR = '../../data/Yelp_train.tfrecords'


def make_tfrecord(input_csv, output_tfr):
    """
    Convert input csv files to standard tensorflow record files. Use float to
    encode every features, use int to encode rate entry.
    """
    table = pd.read_csv(input_csv, header=0).values
    with tf.python_io.TFRecordWriter(MINI_TFR) as tf_writer:
        for row in table:
            # Skip the string columns
            features, rate = np.concatenate((row[4:7], row[11:14])), row[0]
            example = tf.train.Example()
            example.features.feature["features"].float_list.value.extend(
                features)
            # We use float to encode rate labels
            example.features.feature["rate"].int64_list.value.append(rate)
            tf_writer.write(example.SerializeToString())


def read_and_decode(filename_queue):
    """
    Read and decode data from tfrecords file.
    """
    reader = tf.TFRecordReader()
    key, example = reader.read(filename_queue)
    # Parse from the example
    parsed = tf.parse_single_example(
        example,
        # features is a dict matching feature keys
        features={
            'features': tf.FixedLenFeature([], tf.float32),
            'rate': tf.FixedLenFeature([], tf.int64)
        })

    # Decode the result
    features = parsed['features']
    rate = tf.cast(parsed['rate'], tf.int32)

    return features, rate


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)

    # Read the example
    features, rate = read_and_decode(filename_queue)

    # Shuffle the data into batch_size batches
    # How big a buffer we will randomly sample
    min_after_dequeue = 10
    # The max we will pre-batch
    capacity = min_after_dequeue + 3 * batch_size

    # Shuffle and batch
    feature_batch, rate_batch = tf.train.shuffle_batch(
        [features, rate], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return feature_batch, rate_batch


def train():
    """
    Train a one-hidden layer with softmax neural network.
    """
    sess = tf.Session()

    # Build one-hidden-layer computational graph use softmax
    input_hold = tf.placeholder(tf.float32, shape=[None, FEATURE_NUM])
    output_hold = tf.placeholder(tf.float32, shape=[None, OUTPUT_NUM])

    theta = tf.Variable(tf.zeros([FEATURE_NUM, OUTPUT_NUM]))
    bias = tf.Variable(tf.zeros([OUTPUT_NUM]))

    # One-hidden-layer forward
    output_layer = tf.matmul(input_hold, theta) + bias
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=output_hold,
                                                logits=output_layer))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
                                                            cross_entropy)
    
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        # Initialize the queue
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)

        #for _ in range(10):
        #    _, loss_value = sess.run([train_op, loss])
        features, rate = input_pipeline(MINI_TFR, BATCH_SIZE, EPOCH)
        train_step.run(feed_dict={input_hold: features[0], output_hold: rate[0]})


if __name__ == '__main__':
    # I/O Procession, make tfrecords from the training csv file
    # make_tfrecord(MINI_CSV, MINI_TFR)

    # filename_queue = tf.train.string_input_producer(
    #     [MINI_TFR], num_epochs=10)
    # read_and_decode(filename_queue)
    train()
