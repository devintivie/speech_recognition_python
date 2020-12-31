import json
from json import JSONEncoder
from json import JSONDecoder
import numpy as np

def save_network(network):#sizes, weights, biases, filename):
    save_data = {}
    save_data['sizes'] = network.sizes
    # data['biases'] = []
    save_data['biases'] = network.biases
    # data['weights'] = []
    save_data['weights'] = network.weights
    # save_data['input_means'] = network.input_means
    # save_data['input_vars'] = network.input_vars
    with open(network.save_file, 'w') as outfile:
        encodedNumpyData = json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)

def save_lstm_network(network):
    save_data = {}
    save_data['sizes'] = network.sizes
    save_data['weights'] = network.weights
    with open(network.save_file, 'w') as outfile:
        encodedNumpyData = json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)

def load_network_json(filename):
    # Deserialization
    fileString = ''
    with open(filename) as f:
        fileString = f.read()
    network_data = json.loads(fileString)

    # new_s = network_data['sizes']
    new_w = [np.asarray(w) for w in network_data['weights']]
    new_b = [np.asarray(b) for b in network_data['biases']]

    network_data['weights'] = new_w
    network_data['biases'] = new_b
    return network_data
    # nparrays = np.asarray(network_data)
    # return nparrays

def save_zscores(filename, inputs_means, input_vars):
    save_data = {}
    save_data['means'] = inputs_means
    save_data['variances'] = input_vars
    with open(filename, 'w') as outfile:
        encodedNumpyData = json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)

def load_zscores(filename):
    # Deserialization
    fileString = ''
    with open(filename) as f:
        fileString = f.read()
    zscore_data = json.loads(fileString)
    means = zscore_data['means']
    variances = zscore_data['variances']
    return (means, variances)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# class NumpyArrayDecoder(JSONDecoder):
#     def __init__(self, *args, **kwargs):
#         JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

#     def object_hook(self, obj):
#         for 



# import json
# from json import JSONEncoder
# import numpy
# import numpy as np

# # class NumpyArrayEncoder(JSONEncoder):
# #     def default(self, obj):
# #         if isinstance(obj, numpy.ndarray):
# #             return obj.tolist()
# #         return JSONEncoder.default(self, obj)


# # def save_network(sizes, weights, biases, filename):
# #     data = {}
# #     data['input_means'] = 0
# #     data['sizes'] = sizes
# #     # data['biases'] = []
# #     data['biases'] = biases
# #     # data['weights'] = []
# #     data['weights'] =weights
# #     with open(filename, 'w') as outfile:
# #         encodedNumpyData = json.dump(data, outfile, cls=NumpyArrayEncoder, indent=2)


# # def load_network_json(filename):
# #     # Deserialization
# #     fileString = ''
# #     with open(filename) as f:
# #         fileString = f.read()
# #     network_data = json.loads(fileString)

# #     return network_data
#     # new_biases = [np.array(k) for k in network_data['biases']]
#     # new_weights = [np.array(m) for m in network_data['weights']]


#     # return (new_weights, new_biases)



# # # numpyArrayOne = numpy.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
# # sizes = [4,3,2]
# # biases = [np.random.randn(y, 1) for y in sizes[1:]]
# # weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]


# # layer = 1
# # for b,w in biases, weights:
# #     print(f'layer {layer} biases')
# #     print(b)
# #     print(f'layer {layer} weights')
# #     print(w)
# #     layer +=1

# # # Serialization
# # # numpyData = {"array": numpyArrayOne}
# # # biases_data = {"biases": biases}
# # data = {}
# # data['biases'] = []
# # data['biases'].append(biases)
# # data['weights'] = []
# # data['weights'].append(weights)
# # with open('simple_json.json', 'w') as outfile:
# #     encodedNumpyData = json.dump(data, outfile, cls=NumpyArrayEncoder, indent=2)  # use dump() to write array into file
# # print("Printing JSON serialized NumPy array")
# # print(encodedNumpyData)



# # # Deserialization
# # fileString = ''
# # with open('simple_json.json') as f:
# #     fileString = f.read()
# # network_data = json.loads(fileString)

# # new_biases = [np.array(k) for k in network_data['biases'][0] ]
# # new_weights = [np.array(m) for m in network_data['weights'][0] ]

# # print(new_biases)
# # print(type(new_biases))

# # layer = 1
# # for b,w in new_biases, new_weights:
# #     print(f'layer {layer} biases')
# #     print(b)
# #     print(f'layer {layer} weights')
# #     print(w)
# #     layer +=1


# # print("Decode JSON serialized NumPy array")
# # decodedArrays = json.loads(encodedNumpyData)

# # finalNumpyArray = numpy.asarray(decodedArrays["array"])
# # print("NumPy Array")
# # print(finalNumpyArray)