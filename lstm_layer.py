import numpy as np

from network_layer import *
from neural_network_math import *

class lstm_layer():
    def __init__(self, input_length, hidden_length, output_length, direction = 'bidirection'):
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.output_length = output_length
        self.direction = direction

        self.Wxi_fw = neural_weight(hidden_length, input_length)
        self.Whi_fw = neural_weight(hidden_length, hidden_length)
        self.Wci_fw = neural_weight(hidden_length, hidden_length)
        self.Bi_fw = neural_weight(hidden_length, 1)

        self.Wxf_fw = neural_weight(hidden_length, input_length)
        self.Whf_fw = neural_weight(hidden_length, hidden_length)
        self.Wcf_fw = neural_weight(hidden_length, hidden_length)
        self.Bf_fw = neural_weight(hidden_length, 1)

        self.Wxc_fw = neural_weight(hidden_length, input_length)
        self.Whc_fw = neural_weight(hidden_length, hidden_length)
        self.Bc_fw = neural_weight(hidden_length, 1)

        self.Wxo_fw = neural_weight(hidden_length, input_length)
        self.Who_fw = neural_weight(hidden_length, hidden_length)
        self.Bo_fw = neural_weight(hidden_length, 1)

        self.Wy_fw = neural_weight(output_length, hidden_length)
        self.By = neural_weight(output_length, 1)

        if direction == 'bidirection':
            self.Wxi_rv = neural_weight(hidden_length, input_length)
            self.Whi_rv = neural_weight(hidden_length, hidden_length)
            self.Wci_rv = neural_weight(hidden_length, hidden_length)
            self.Bi_rv = neural_weight(hidden_length, 1)

            self.Wxf_rv = neural_weight(hidden_length, input_length)
            self.Whf_rv = neural_weight(hidden_length, hidden_length)
            self.Wcf_rv = neural_weight(hidden_length, hidden_length)
            self.Bf_rv = neural_weight(hidden_length, 1)

            self.Wxc_rv = neural_weight(hidden_length, input_length)
            self.Whc_rv = neural_weight(hidden_length, hidden_length)
            self.Bc_rv = neural_weight(hidden_length, 1)

            self.Wxo_rv = neural_weight(hidden_length, input_length)
            self.Who_rv = neural_weight(hidden_length, hidden_length)
            self.Bo_rv = neural_weight(hidden_length, 1)

            self.Wy_rv = neural_weight(output_length, hidden_length)

    def layer_forward(self, activation):
        # self.all_activations_fws = []
        # self.ai_fw = []
        # self.af_fw = []
        # self.ac_fw = []
        # self.ao_fw = []

        # self.all_c_fws = []  #all ct forward status
        # self.all_c_prev_fws = []     #all ct-1 forward status

        # self.all_h_fws = []    #all ht forward activations
        # self.all_h_prev_fws = []    #all ht-1 forward activations 

        self.forward_activation = activation
        self.compute_forward_sequence()

        if self.direction == 'bidirection':
            # self.all_activations_rvs = []
            
            # self.all_c_rvs = []  #all ct reverse status
            # self.all_c_prev_rvs = []     #all ct-1 reverse status

            # self.all_h_rvs = []
            # self.all_h_prev_rvs = []
            # self.ai_rv = []
            # self.af_rv = []
            # self.ac_rv = []
            # self.ao_rv = []

            self.reverse_activation = np.flip(activation)
            self.compute_reverse_sequence()

        
        
        return self.compute_output()

    def compute_forward_sequence(self):
        cell = np.zeros((self.hidden_length, 1))
        hidden = np.zeros((self.hidden_length, 1))        
        cells = [cell]
        hiddens = [hidden]

        ai_list = []        #input gate sum
        af_list = []        #forward gate sum
        ao_list = []        #output gate sum
        ac_list = []        #cell gate sum

        for Xt in self.forward_activation:
            Xt = np.atleast_2d(Xt).T

            input_sum = np.dot(self.Wxi_fw.weight, Xt) + np.dot(self.Whi_fw.weight, hidden) + np.dot(self.Wci_fw.weight, cell) + self.Bi_fw.weight
            input_gate = sigmoid(input_sum)
            ai_list.append(input_sum)

            forget_sum = np.dot(self.Wxf_fw.weight, Xt) + np.dot(self.Whf_fw.weight, hidden) + np.dot(self.Wcf_fw.weight, cell) + self.Bf_fw.weight
            forget_gate = sigmoid(forget_sum)
            af_list.append(forget_sum)

            ac = np.dot(self.Wxc_fw.weight, Xt) + np.dot(self.Whc_fw.weight, hidden) + self.Bc_fw.weight
            cell = forget_gate * cell + input_gate * np.tanh(ac)
            ac_list.append(cell)

            output_sum = np.dot(self.Wxo_fw.weight, Xt) + np.dot(self.Who_fw.weight, hidden) + self.Bo_fw.weight 
            output_gate = sigmoid(output_sum)
            ao_list.append(output_sum)

            hidden = output_gate * np.tanh(cell)

            cells.append(cell)
            hiddens.append(hidden)

        self.c_fw = np.concatenate(cells[1:], axis=1) #all ct forward status
        self.c_fw_prev = np.concatenate(cells[:-1], axis=1)     #all ct-1 forward status

        self.ai_fw = np.concatenate(ai_list, axis=1)
        self.af_fw = np.concatenate(af_list, axis=1)
        self.ac_fw = np.concatenate(ac_list, axis=1)
        self.ao_fw = np.concatenate(ao_list, axis=1)

        self.h_fw = np.concatenate(hiddens[1:], axis=1)    #all ht forward activations
        self.h_fw_prev = np.concatenate(hiddens[:-1], axis=1)    #all ht-1 forward activations 

        # self.all_activations_fws.append(self.all_h_fws[-1]) #get first = last = only, will change when multiple layers are used
        # self.current_activation_fw = self.all_h_fws[-1]

    def compute_reverse_sequence(self):
        cell = np.zeros((self.hidden_length, 1))
        hidden = np.zeros((self.hidden_length, 1))        
        cells = [cell]
        hiddens = [hidden]

        ai_list = []        #input gate sum
        af_list = []        #forward gate sum
        ao_list = []        #output gate sum
        ac_list = []        #cell gate sum

        for Xt in self.reverse_activation:#self.current_activation.T:
            Xt = np.atleast_2d(Xt).T

            input_sum = np.dot(self.Wxi_rv.weight, Xt) + np.dot(self.Whi_rv.weight, hidden) + np.dot(self.Wci_rv.weight, cell) + self.Bi_rv.weight
            input_gate = sigmoid(input_sum)
            ai_list.append(input_sum)

            forget_sum = np.dot(self.Wxf_rv.weight, Xt) + np.dot(self.Whf_rv.weight, hidden) + np.dot(self.Wcf_rv.weight, cell) + self.Bf_rv.weight
            forget_gate = sigmoid(forget_sum)
            af_list.append(forget_sum)

            ac = np.dot(self.Wxc_rv.weight, Xt) + np.dot(self.Whc_rv.weight, hidden) + self.Bc_rv.weight
            cell = forget_gate * cell + input_gate * np.tanh(ac)
            ac_list.append(cell)

            output_sum = np.dot(self.Wxo_rv.weight, Xt) + np.dot(self.Who_rv.weight, hidden) + self.Bo_rv.weight
            output_gate = sigmoid(output_sum)
            ao_list.append(output_sum)

            hidden = output_gate * np.tanh(cell)

            cells.append(cell)
            hiddens.append(hidden)

        cells = np.flip(cells)
        hiddens = np.flip(hiddens)

        self.c_rv = np.concatenate(cells[1:], axis=1) #all ct forward status
        self.c_rv_prev = np.concatenate(cells[:-1], axis=1)     #all ct-1 forward status

        self.ai_rv = np.concatenate(ai_list, axis=1)
        self.af_rv = np.concatenate(af_list, axis=1)
        self.ac_rv = np.concatenate(ac_list, axis=1)
        self.ao_rv = np.concatenate(ao_list, axis=1)

        self.h_rv = np.concatenate(hiddens[1:], axis=1)    #all ht forward activations
        self.h_rv_prev = np.concatenate(hiddens[:-1], axis=1)    #all ht-1 forward activations 

        # self.all_activations_rvs.append(self.all_h_rvs[-1]) #get first = last = only, will change when multiple layers are used
        # self.current_activation_rv = self.h_rvs



    def compute_output(self):
        self.output = np.dot(self.Wy_fw.weight, self.h_fw) + self.By.weight
        if self.direction == 'bidirection':
            reverse = np.dot(self.Wy_rv.weight, self.h_rv)
            self.output += reverse
        
        return self.output


