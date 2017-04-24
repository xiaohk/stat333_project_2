import numpy as np
import pickle


# Read the glove dictionary
with open('glove.6B.100d.txt', 'r') as fp:
    embedding_dict = {}
    for line in fp.readlines():
        splitted = line.split()
        embedding_dict[splitted[0]] = np.array(splitted[1:], dtype='float32')

# Pickle the dictionary
with open('glove100d.pickle', 'wb') as fp:
    pickle.dump(embedding_dict, fp)


# test
with open('glove100d.pickle', 'rb') as fp:
    my_dict = pickle.load(fp)
    print(type(my_dict))
    print(type(my_dict['apple']))
    print(my_dict['apple'])
    print(my_dict['apple'].dtype)
