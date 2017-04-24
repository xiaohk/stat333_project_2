from pickle import load
from scipy.io import mmwrite


def move_matrix(input_fn, output_fn, comment=""):
    """ Convert scipy parse matrix to R matrix, using MatrixMarket format.
        `input_fn` and `output_fn` are two file names. `input_fn` should be
        a python pickle file.
    """
    matrix = load(open(input_fn, 'rb'))
    mmwrite(output_fn, matrix, comment)


if __name__ == '__main__':
    # Full matrix
    move_matrix('../../static/tf_matrix.pickle', './config/matrix.mtx')
    # Lasso selected matrix
    move_matrix('../../static/selected_matrix.pickle',
                './config/selected_matrix.mtx')
