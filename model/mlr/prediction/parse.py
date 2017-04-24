def add_id(input_csv, output_csv):
    with open(input_csv, 'r') as in_fp:
        with open(output_csv, 'w') as out_fp:
            # Skip the fist line
            in_fp.readline()
            out_fp.write('"Id", "Prediction"\n')
            cur_id = 1
            for line in in_fp.readlines():
                out_fp.write("{}, {}".format(cur_id, line))
                cur_id += 1


add_id('prediction_old.csv', 'prediction_1.csv')
