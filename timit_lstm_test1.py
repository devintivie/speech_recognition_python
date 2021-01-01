
from neural_network import neural_network
import numpy as np
from data_save_and_load import *

from neural_network_math import *
from neural_network_file import *

filename = 'mfcc39_sentences_features_DR1.json'
training_data = load_training_sentences(filename) 
training_inputs = training_data[0]
training_outputs = training_data[1]

test_data = load_test_sentences(filename) 
test_inputs = test_data[0]
test_outputs = test_data[1]


print('training and test data loaded')


zscore_filename = 'timit_lstm_zscores.json'
save_zscore = False
if save_zscore:
    input_means, input_vars = calculate_zscore_sent(training_inputs)
    save_zscores(zscore_filename, input_means, input_vars)
    
    print('zscores save updated')
load_means, load_vars = load_zscores(zscore_filename)
print('zscores loaded')

input_data = preprocess_zscore(training_inputs, load_means, load_vars)
test_inputs = preprocess_zscore(test_inputs, load_means, load_vars)

print('zscores processed')


# network = neural_network([39, 40, 20, 40, 30, 61])
network = neural_network()
network.add_layer(39, 40, 30)
network.add_layer(30, 30, 30)
network.add_layer(30, 30, 30)
network.add_layer(30, 30, 30)
network.add_layer(30, 30, 61)
learn_rate = 1e-4
momentum = 0.85
penalty = 0.35
# batch_size = 100


test_data = list(zip(test_inputs, test_outputs))

# input_data = [input_data[0]]
# training_outputs = [training_outputs[0]]
# network.save_file = 'timit_lstm_network_1e-4_85_40_40_40_40.json'
max_norm = 1
network.train(input_data, training_outputs, learn_rate, momentum, 100, penalty, test_data)

# i = 0
# for t in test_data:
#     if i == 5:
#         break
#     values = network.test(t[0].T)[1]
#     selection = network.test(t[0].T)[0]
#     print(f"test {i+1}: {values}") # 0.951 - F
#     print(f"test {i+1}: {selection} = {t[1]}")
# network.train(training_data)
# net = single_layer_neural_network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


