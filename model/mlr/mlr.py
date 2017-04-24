def predict(estimator, all_matrix, train_length, output_csv):
    """ Predict the test and validation text, and write to csv. The estimator
        should be a prediction estimator, instead of a classifier.
    """
    # Read the text
    # `all_matrix` has already contained all the test and vali text
    x_predict = all_matrix[train_length:]
    print("Successfully load predicting text, with shape {}.".format(
        x_predict.shape))

    prediction = estimator.predict(x_predict)

    # Combine ID and write to a file
    with open(output_csv, 'w') as output:
        output.write('"Id","Prediction"\n')
        for i in range(len(prediction)):
            output.write("{},{}\n".format(i, prediction[i]))
