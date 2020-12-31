import numpy as np
import json
from neural_network_file import NumpyArrayEncoder

from phoneme_mapping import map_phoneme_to_output_array, phoneme_rev, phonemes

def save_data(filename, tr_d = None, te_d = None, va_d = None):
    save_data = {}
    save_data['tr_d'] = tr_d
    save_data['te_d'] = te_d
    save_data['va_d'] = va_d
    with open(filename, 'w') as outfile:
        json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)

def combine_data(input_data, output_data):
    data = []
    data.append(input_data)
    data.append(output_data)
    return data

# def save_training_data(filename, input_matrix, output_array):
#     save_data = {}
#     save_data['tr_d'] = []
#     save_data['tr_d'].append(input_matrix)
#     save_data['tr_d'].append(output_array)
#     with open(filename, 'w') as outfile:
#         json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)

# def save_test_data(filename, test_inputs, test_outputs):
#     save_data = {}
#     save_data['te_d'] = []
#     save_data['te_d'].append(test_inputs)
#     save_data['te_d'].append(test_outputs)
#     with open(filename, 'w') as outfile:
        # json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)

def load_training_data(filename):
    # Deserialization
    fileString = ''
    with open(filename) as f:
        fileString = f.read()
    training_data = json.loads(fileString)['tr_d']

    training_samples = len(training_data[0])
    input_array_length = len(training_data[0][0])
    decisions_length = 61
    
    training_inputs = np.zeros((training_samples, input_array_length))
    training_outputs = np.zeros((training_samples, decisions_length))
    i = 0
    for x in training_data[0]:
        training_inputs[i, :] = x#np.reshape(x, (input_array_length, 1))
        i += 1

    i = 0
    # print(type(training_data[1]))
    for y in training_data[1]:
        training_outputs[i, :] = map_phoneme_to_output_array(y)
        i += 1

    return (training_inputs, training_outputs)

def load_training_sentences(filename):
    # Deserialization
    fileString = ''
    with open(filename) as f:
        fileString = f.read()
    training_data_s, training_data_y = json.loads(fileString)['tr_d']
    training_data = list(zip(training_data_s, training_data_y))
    training_inputs = []
    training_outputs = []
    for d in training_data:
        training_samples = len(d[0])
        input_array_length = len(d[0][0])
        decisions_length = 61
        
        sentence_inputs = np.zeros((training_samples, input_array_length))
        sentence_outputs = np.zeros((training_samples, decisions_length))
        i = 0
        for x in d[0]:
            sentence_inputs[i, :] = x#np.reshape(x, (input_array_length, 1))
            i += 1

        i = 0
        # print(type(d[1]))
        for y in d[1]:
            sentence_outputs[i, :] = map_phoneme_to_output_array(y)
            i += 1

        training_inputs.append(sentence_inputs)
        training_outputs.append(sentence_outputs)
        

    return (training_inputs, training_outputs)

def load_test_data(filename):
    # Deserialization
    fileString = ''
    with open(filename) as f:
        fileString = f.read()
    testing_data = json.loads(fileString)['te_d']

    test_samples = len(testing_data[0])
    input_array_length = len(testing_data[0][0])
    decisions_length = 61
    
    testing_inputs = np.zeros((test_samples, input_array_length))
    # testing_outputs = np.zeros((test_samples, decisions_length))
    testing_outputs = []
    i = 0
    for x in testing_data[0]:
        testing_inputs[i, :] = x#np.reshape(x, (input_array_length, 1))

        i += 1

    i = 0
    # print(type(testing_data[1]))
    for y in testing_data[1]:
        testing_outputs.append(phonemes[y])
        #  testing_outputs[i, :] = map_phoneme_to_output_array(y)
        i += 1


    return (testing_inputs, testing_outputs)


def load_test_sentences(filename):
    # Deserialization
    fileString = ''
    with open(filename) as f:
        fileString = f.read()
    testing_data_s, testing_data_y = json.loads(fileString)['te_d']
    testing_data = list(zip(testing_data_s, testing_data_y))
    testing_inputs = []
    testing_outputs = []

    for t in testing_data:
        test_samples = len(t[0])
        input_array_length = len(t[0][0])
        decisions_length = 61
        
        sentence_inputs = np.zeros((test_samples, input_array_length))
        # testing_outputs = np.zeros((test_samples, decisions_length))
        sentence_outputs = []
        i = 0
        for x in t[0]:
            sentence_inputs[i, :] = x#np.reshape(x, (input_array_length, 1))

            i += 1

        i = 0
        # print(type(t[1]))
        for y in t[1]:
            sentence_outputs.append(phonemes[y])
            #  sentence_outputs[i, :] = map_phoneme_to_output_array(y)
            i += 1

        testing_inputs.append(sentence_inputs)
        testing_outputs.append(sentence_outputs)

    return (testing_inputs, testing_outputs)

if __name__ == "__main__":
    # x = np.array([[1, 2, 3 ,3, 1], [2, 2, 2, 2,2 ], [5, 4, 3, 2, 1]])
    # y = ['iy', 'l', 'ix']

    # filename = 'numpy_test.json'

    
    # save_training_data(filename, x, y)
    filename = f"mfcc_features_DR1.json"

    training_data = load_training_data(filename)

    print()


