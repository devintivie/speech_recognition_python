import numpy as np

from network_layer import *
from neural_network_math import *

class lstm_layer():
    def __init__(self, input_length, hidden_length, output_length, direction = 'bidirection'):
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.output_length = output_length
        # self.direction = direction

        self.init_forward_weights()

        if direction == 'bidirection':
            self.bidirectional = True
            self.init_reverse_weights()
        else:
            self.bidirectional = False
            
    def init_forward_weights(self):
        self.Wxi_fw = neural_weight(self.hidden_length, self.input_length)
        self.Whi_fw = neural_weight(self.hidden_length, self.hidden_length)
        self.Wci_fw = neural_weight(self.hidden_length, self.hidden_length)
        self.Bi_fw = neural_weight(self.hidden_length, 1, 'zero')

        self.Wxf_fw = neural_weight(self.hidden_length, self.input_length)
        self.Whf_fw = neural_weight(self.hidden_length, self.hidden_length)
        self.Wcf_fw = neural_weight(self.hidden_length, self.hidden_length)
        self.Bf_fw = neural_weight(self.hidden_length, 1, 'zero')

        self.Wxc_fw = neural_weight(self.hidden_length, self.input_length)
        self.Whc_fw = neural_weight(self.hidden_length, self.hidden_length)
        self.Bc_fw = neural_weight(self.hidden_length, 1, 'zero')

        self.Wxo_fw = neural_weight(self.hidden_length, self.input_length)
        self.Who_fw = neural_weight(self.hidden_length, self.hidden_length)
        self.Bo_fw = neural_weight(self.hidden_length, 1, 'zero')

        self.Wy_fw = neural_weight(self.output_length, self.hidden_length)
        self.By = neural_weight(self.output_length, 1, 'zero')

    def init_reverse_weights(self):
        self.Wxi_rv = neural_weight(self.hidden_length, self.input_length)
        self.Whi_rv = neural_weight(self.hidden_length, self.hidden_length)
        self.Wci_rv = neural_weight(self.hidden_length, self.hidden_length)
        self.Bi_rv = neural_weight(self.hidden_length, 1)

        self.Wxf_rv = neural_weight(self.hidden_length, self.input_length)
        self.Whf_rv = neural_weight(self.hidden_length, self.hidden_length)
        self.Wcf_rv = neural_weight(self.hidden_length, self.hidden_length)
        self.Bf_rv = neural_weight(self.hidden_length, 1)

        self.Wxc_rv = neural_weight(self.hidden_length, self.input_length)
        self.Whc_rv = neural_weight(self.hidden_length, self.hidden_length)
        self.Bc_rv = neural_weight(self.hidden_length, 1)

        self.Wxo_rv = neural_weight(self.hidden_length, self.input_length)
        self.Who_rv = neural_weight(self.hidden_length, self.hidden_length)
        self.Bo_rv = neural_weight(self.hidden_length, 1)

        self.Wy_rv = neural_weight(self.output_length, self.hidden_length)

    def layer_forward(self, activation):
        self.layer_input = activation.T
        self.forward_activation = activation
        self.compute_forward_sequence()

        if self.bidirectional:
            self.reverse_activation = np.flip(activation)
            self.compute_reverse_sequence()

        return self.compute_output()

    def layer_backprop(self, d_activation):
        num_samples = d_activation.shape[1]
        self.averager = 1.0/num_samples
        dL_dht = self.calc_dY(d_activation)
        dL_dao = self.calc_dO(dL_dht)
        (dL_dct, dL_dac) = self.calc_dC(dL_dht)
        dL_daf = self.calc_dF(dL_dct)
        dL_dai = self.calc_dI(dL_dct)
        dL_dzt = self.calc_dInputs(dL_daf, dL_dai, dL_dao, dL_dac)
        return dL_dzt

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

    def compute_output(self):
        self.output = np.dot(self.Wy_fw.weight, self.h_fw) + self.By.weight
        if self.bidirectional:
            reverse = np.dot(self.Wy_rv.weight, self.h_rv)
            self.output += reverse
        
        self.output = softmax(self.output)
        return self.output

    def calc_dY(self, d_activation):
        self.Wy_fw.delta = self.averager * np.dot(d_activation, self.h_fw.T)
        self.By.delta = np.atleast_2d(np.average(d_activation, 1)).T
        dL_dht_fw = np.dot(self.Wy_fw.weight.T, d_activation)

        if self.bidirectional:
            self.Wy_rv.delta = self.averager * np.dot(d_activation, self.h_rv.T)
            dL_dht_rv = np.dot(self.Wy_rv.weight.T, d_activation)
        else:
            dL_dht_rv = None
        return (dL_dht_fw, dL_dht_rv)

    def calc_dO(self, dL_dht):
        dL_dht_fw = dL_dht[0]
        dL_dht_rv = dL_dht[1]

        dL_dot_fw = dL_dht_fw * np.tanh(self.c_fw)
        dL_dao_fw = dL_dot_fw * sigmoid_prime(self.ao_fw)

        self.Wxo_fw.delta = self.averager * np.dot(dL_dao_fw, self.layer_input.T)
        self.Who_fw.delta = self.averager * np.dot(dL_dao_fw, self.h_fw_prev.T)
        self.Bo_fw.delta = np.atleast_2d(np.average(dL_dot_fw, axis=1)).T

        if self.bidirectional:
            dL_dot_rv = dL_dht_rv * np.tanh(self.c_rv)
            dL_dao_rv = dL_dot_rv * sigmoid_prime(self.ao_rv)

            self.Wxo_rv.delta = self.averager * np.dot(dL_dao_rv, self.layer_input.T)
            self.Who_rv.delta = self.averager * np.dot(dL_dao_rv, self.h_rv_prev.T)
            self.Bo_rv.delta = np.atleast_2d(np.average(dL_dot_rv, axis=1)).T
        else:
            dL_dao_rv = None
        return (dL_dao_fw, dL_dao_rv)

    def calc_dC(self, dL_dht):
        dL_dht_fw = dL_dht[0]
        dL_dht_rv = dL_dht[1]

        #dL_dc hat
        dL_dct_fw = dL_dht_fw * sigmoid(self.ao_fw) * (1.0 - np.power( np.tanh(self.c_fw), 2 ))
        dL_dcet_fw = dL_dct_fw * sigmoid(self.ai_fw)
        dL_dac_fw = dL_dcet_fw * (1.0 - np.power( np.tanh(self.ac_fw), 2 ))

        self.Wxc_fw.delta = self.averager * np.dot(dL_dac_fw, self.layer_input.T)
        self.Whc_fw.delta = self.averager * np.dot(dL_dac_fw, self.h_fw_prev.T)
        self.Bc_fw.delta = np.atleast_2d(np.average(dL_dac_fw, axis=1)).T

        if self.bidirectional:
            dL_dct_rv = dL_dht_rv * sigmoid(self.ao_rv) * (1.0 - np.power( np.tanh(self.c_rv), 2 ))
            dL_dcet_rv = dL_dct_rv * sigmoid(self.ai_rv)
            dL_dac_rv = dL_dcet_rv * (1.0 - np.power( np.tanh(self.ac_rv), 2 ))

            self.Wxc_rv.delta = self.averager * np.dot(dL_dac_rv, self.layer_input.T)
            self.Whc_rv.delta = self.averager * np.dot(dL_dac_rv, self.h_rv_prev.T)
            self.Bc_rv.delta = np.atleast_2d(np.average(dL_dac_rv, axis=1)).T
        else:
            dL_dct_rv = None
            dL_dac_rv = None

        dL_dct = (dL_dct_fw, dL_dct_rv)    
        dL_dac = (dL_dac_fw, dL_dac_rv)
        return (dL_dct, dL_dac)

    def calc_dF(self, dL_dct):
        dL_dct_fw = dL_dct[0]
        dL_dct_rv = dL_dct[1]

        dL_dft_fw = dL_dct_fw * self.c_fw_prev
        dL_daf_fw = dL_dft_fw * sigmoid_prime(self.af_fw)

        self.Wxf_fw.delta = np.dot(dL_daf_fw, self.layer_input.T)
        self.Whf_fw.delta = np.dot(dL_daf_fw, self.h_fw.T)
        self.Wcf_fw.delta = np.dot(dL_daf_fw, self.c_fw.T)
        self.Bf_fw.delta = np.atleast_2d(np.average(dL_daf_fw, axis=1)).T

        if self.bidirectional:
            dL_dft_rv = dL_dct_rv * self.c_rv_prev
            dL_daf_rv = dL_dft_rv * sigmoid_prime(self.af_rv)

            self.Wxf_rv.delta = np.dot(dL_daf_rv, self.layer_input.T)
            self.Whf_rv.delta = np.dot(dL_daf_rv, self.h_rv.T)
            self.Wcf_rv.delta = np.dot(dL_daf_rv, self.c_rv.T)
            self.Bf_rv.delta = np.atleast_2d(np.average(dL_daf_rv, axis=1)).T
        else:
            dL_daf_rv = None
        
        return (dL_daf_fw, dL_daf_rv)

    def calc_dI(self, dL_dct):
        dL_dct_fw = dL_dct[0]
        dL_dct_rv = dL_dct[1]

        dL_dit_fw = dL_dct_fw * np.tanh(self.ac_fw)
        dL_dai_fw = dL_dit_fw * sigmoid_prime(self.ai_fw)

        self.Wxi_fw.delta = np.dot(dL_dai_fw, self.layer_input.T)
        self.Whi_fw.delta = np.dot(dL_dai_fw, self.h_fw.T)
        self.Wci_fw.delta = np.dot(dL_dai_fw, self.c_fw.T)
        self.Bi_fw.delta = np.atleast_2d(np.average(dL_dai_fw, axis=1)).T

        if self.bidirectional:
            dL_dit_rv = dL_dct_rv * np.tanh(self.ac_rv)
            dL_dai_rv = dL_dit_rv * sigmoid_prime(self.ai_rv)

            self.Wxi_rv.delta = np.dot(dL_dai_rv, self.layer_input.T)
            self.Whi_rv.delta = np.dot(dL_dai_rv, self.h_rv.T)
            self.Wci_rv.delta = np.dot(dL_dai_rv, self.c_rv.T)
            self.Bi_rv.delta = np.atleast_2d(np.average(dL_dai_rv, axis=1)).T
        else:
            dL_dai_rv = None
        
        return (dL_dai_fw, dL_dai_rv)

    def calc_dInputs(self, dL_daf, dL_dai, dL_dao, dL_dac):
        dL_daf_fw = dL_daf[0]
        dL_daf_rv = dL_daf[1]

        dL_dai_fw = dL_dai[0]
        dL_dai_rv = dL_dai[1]

        dL_dao_fw = dL_dao[0]
        dL_dao_rv = dL_dao[1]

        dL_dac_fw = dL_dac[0]
        dL_dac_rv = dL_dac[1]

        dl_dzt = np.dot(self.Wxf_fw.weight.T, dL_daf_fw)
        dl_dzt += np.dot(self.Wxi_fw.weight.T, dL_dai_fw)
        dl_dzt += np.dot(self.Wxo_fw.weight.T, dL_dao_fw)
        dl_dzt += np.dot(self.Wxc_fw.weight.T, dL_dac_fw)

        if self.bidirectional:
            dl_dzt += np.dot(self.Wxf_rv.weight.T, dL_daf_rv)
            dl_dzt += np.dot(self.Wxi_rv.weight.T, dL_dai_rv)
            dl_dzt += np.dot(self.Wxo_rv.weight.T, dL_dao_rv)
            dl_dzt += np.dot(self.Wxc_rv.weight.T, dL_dac_rv)

        return dl_dzt

    def update_weights(self, learn_rate, momentum):
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.update_forward_velocities()
        if self.bidirectional:
            self.update_reverse_velocities()

        self.update_forward_weights()
        if self.bidirectional:
            self.update_reverse_weights()

    def update_forward_velocities(self):
        self.update_velocity(self.Wxi_fw)
        self.update_velocity(self.Whi_fw)
        self.update_velocity(self.Wci_fw)
        self.update_velocity(self.Bi_fw)

        self.update_velocity(self.Wxf_fw)
        self.update_velocity(self.Whf_fw)
        self.update_velocity(self.Wcf_fw)
        self.update_velocity(self.Bf_fw)

        self.update_velocity(self.Wxc_fw)
        self.update_velocity(self.Whc_fw)
        self.update_velocity(self.Bc_fw)

        self.update_velocity(self.Wxo_fw)
        self.update_velocity(self.Who_fw)
        self.update_velocity(self.Bo_fw)

        self.update_velocity(self.Wy_fw)
        self.update_velocity(self.By)

    def update_reverse_velocities(self):
        self.update_velocity(self.Wxi_rv)
        self.update_velocity(self.Whi_rv)
        self.update_velocity(self.Wci_rv)
        self.update_velocity(self.Bi_rv)

        self.update_velocity(self.Wxf_rv)
        self.update_velocity(self.Whf_rv)
        self.update_velocity(self.Wcf_rv)
        self.update_velocity(self.Bf_rv)

        self.update_velocity(self.Wxc_rv)
        self.update_velocity(self.Whc_rv)
        self.update_velocity(self.Bc_rv)

        self.update_velocity(self.Wxo_rv)
        self.update_velocity(self.Who_rv)
        self.update_velocity(self.Bo_rv)

        self.update_velocity(self.Wy_rv)

    def update_forward_weights(self):
        self.update_weight(self.Wxi_fw)
        self.update_weight(self.Whi_fw)
        self.update_weight(self.Wci_fw)
        self.update_weight(self.Bi_fw)

        self.update_weight(self.Wxf_fw)
        self.update_weight(self.Whf_fw)
        self.update_weight(self.Wcf_fw)
        self.update_weight(self.Bf_fw)

        self.update_weight(self.Wxc_fw)
        self.update_weight(self.Whc_fw)
        self.update_weight(self.Bc_fw)

        self.update_weight(self.Wxo_fw)
        self.update_weight(self.Who_fw)
        self.update_weight(self.Bo_fw)

        self.update_weight(self.Wy_fw)
        self.update_weight(self.By)

    def update_reverse_weights(self):
        self.update_weight(self.Wxi_rv)
        self.update_weight(self.Whi_rv)
        self.update_weight(self.Wci_rv)
        self.update_weight(self.Bi_rv)

        self.update_weight(self.Wxf_rv)
        self.update_weight(self.Whf_rv)
        self.update_weight(self.Wcf_rv)
        self.update_weight(self.Bf_rv)

        self.update_weight(self.Wxc_rv)
        self.update_weight(self.Whc_rv)
        self.update_weight(self.Bc_rv)

        self.update_weight(self.Wxo_rv)
        self.update_weight(self.Who_rv)
        self.update_weight(self.Bo_rv)

        self.update_weight(self.Wy_rv)

    def update_velocity(self, weight):
        weight.velocity = self.momentum * weight.velocity + self.learn_rate * weight.delta

    def update_weight(self, weight):
        weight.weight = weight.weight - weight.velocity
        weight_max = np.max(weight.weight)
        weight_min = np.min(weight.weight)

        # weight = softmax(weight)
        
        # if weight_max > 3 :
        #     print(f"weight max over threshold at {weight_max}")

        # if weight_max < -3 :
        #     print(f"weight min is under threshold at {weight_min}")
