import numpy as np
from enum import Enum

from np_helpers import vectorized_result

def map_phoneme_to_output_array(phoneme_str):
    y = phonemes[phoneme_str]
    
    output_array = vectorized_result(y, len(phonemes))
    return output_array

def map_num_to_phoneme(value):
    return phoneme_rev(value)



phonemes = {
    'iy':0,
    'ih':1,
    'eh':2,
    'ey':3,
    'ae':4,
    'aa':5,
    'aw':6,
    'ay':7,
    'ah':8,
    'ao':9,
    'oy':10,
    'ow':11,
    'uh':12,
    'uw':13,
    'ux':14,
    'er':15,
    'ax':16,
    'ix':17,
    'axr':18,
    'axh':19,
    'jh':20,
    'ch':21,
    'b':22,
    'd':23,
    'g':24,
    'p':25,
    't':26,
    'k':27,
    'dx':28,
    's':29,
    'sh':30,
    'z':31,
    'zh':32,
    'f':33,
    'th':34,
    'v':35,
    'dh':36,
    'm':37,
    'n':38,
    'ng':39,
    'em':40,
    'nx':41,
    'en':42,
    'eng':43,
    'l':44,
    'r':45,
    'w':46,
    'y':47,
    'hh':48,
    'hv':49,
    'el':50,
    'bcl':51,
    'dcl':52,
    'gcl':53,
    'pcl':54,
    'tcl':55,
    'kcl':56,
    'q':57,
    'pau':58,
    'epi':59,
    'h#':60
}

phoneme_rev = {v: k for k,v in phonemes.items()}


if __name__ == "__main__":
    print(phoneme_rev.get('iy'))
    print(phonemes[0])