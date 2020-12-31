import numpy as np
import random
import json
from json import JSONEncoder

from neural_network_math import *
from neural_network_file import *

from lstm_layer import *
import time

np.set_printoptions(precision=4, suppress=True)


class neural_network:
    def __init__(self):
        self.layers = []
        self.layer_count = 0
        print('network initialized')

    def add_layer(self, input_length, hidden_length, activation_length, layer_type = 'lstm'):
        new_layer = lstm_layer(input_length=input_length, hidden_length=hidden_length, output_length=activation_length)
        self.layers.append(new_layer)
        self.layer_count += 1
        print(f'layer added as layer{self.layer_count}')

    def feed_forward(self, batch_inputs):
        self.all_activations = []
        self.current_activation = batch_inputs.T
        self.all_activations.append(self.current_activation)

        for i in range(self.layer_count):
            ay = self.layers[i].layer_forward(self.current_activation.T)
            self.current_activation = ay
            self.all_activations.append(ay)

        self.y_hat = self.current_activation
        return self.y_hat

    def train(self, data, all_y_trues, learn_rate, momentum, epochs, test_data = None):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        n = len(data)
        self.learn_rate = learn_rate
        self.momentum = momentum

        #data needs to stay sequential so do not randomize       
        for epoch in range(epochs):
            print(epoch)
            correct_count = 0
            total_count = 0
            shuffle_data = list(zip(data, all_y_trues))
            random.shuffle(shuffle_data)

            for sentence_inputs,sentence_trues in shuffle_data:
                predicts = self.feed_forward(sentence_inputs)
                result = np.argmax(predicts, axis=0)
                trues = np.argmax(sentence_trues.T, axis=0)
                total_count += len(result)
                correct_count += np.count_nonzero(result == trues)
                self.backprop(sentence_trues)

            percent_correct = correct_count/total_count*100
            print(f"percent correct = {percent_correct}")
            print()

            # if self.save_file != None :
            #     save_lstm_network(self)

            # if test_data != None:
            #     self.test(test_data)


    





if __name__ == "__main__":
    network = neural_network()
    network.add_layer(39, 40, 61)

    