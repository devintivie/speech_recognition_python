
from timit_loader import *

import math
import chunk_data
import soundfile as sf
import numpy as np
from scipy.fftpack import dct

import json
from neural_network_file import NumpyArrayEncoder
# from numpy_file_testing import save_training_data, save_test_data
from data_save_and_load import save_data, combine_data

region_to_process = 'DR1'
training_files = get_files_per_region('TRAIN', region_to_process)
test_files = get_files_per_region('TEST', region_to_process)
# filenames = get_all_file_prefixes()

frame_len = 20e-3
jump_len = 10e-3
# training_data = np.empty((1,12))
training_data = []
test_data = []
training_sounds = []
test_sounds = []

# def save_region_data(region_str, feature_matrix):
#     save_data = {}
#     save_data['tr_d'] = feature_matrix
#     with open(f"mfcc_features_{region_str}.json", 'w') as outfile:
#         json.dump(save_data, outfile, cls=NumpyArrayEncoder, indent=2)
files_to_process = [training_files, test_files]

for purpose in files_to_process:
    #all files in either training or testing directories
    for f in purpose:
        train_sentence_data = []
        train_sentence_sounds = []
        test_sentence_data = []
        test_sentence_sounds = []
        

        print(f)
        print(f"length of training_data = {len(training_data)}")
        print(f"length of training_sounds = {len(training_sounds)}")
        print(f"length of test_data = {len(test_data)}")
        print(f"length of test_sounds = {len(test_sounds)}")
        sounds = list()
        
        with open(f"{f}.PHN") as file:
            for row in file:
                start, stop, phoneme = row.split()

                data = phn(start, stop, phoneme)
                sounds.append(data)

        data, fs = sf.read(f"{f}.wav")

        N = int(fs * frame_len) #320
        K = fs * jump_len #160
        T = N/fs
        dt = 1/fs
        t = np.arange(0., T, dt)
        max_freq_mel = 1127*np.log(1+(fs/2)/700)
        nfilt = 26
        mel_points = np.linspace(0,max_freq_mel, nfilt+2)
        mel_to_hz = 700*(np.e**(mel_points/1127)-1)

        fbank = np.zeros((nfilt, int(np.floor(N/2+1))))

        # mel_to_hz = (mel_points/1127)
        # print(mel_to_hz)

        #mel bin start and end points based on fft index
        bins = np.floor((N + 1) * mel_to_hz / fs)
        # print(bins)

        for m in range(1, nfilt + 1):
            f_m_left = int(bins[m-1])
            f_m_middle = int(bins[m])
            f_m_right = int(bins[m+1])

            for k in range(f_m_left, f_m_middle):
                num = (k-bins[m-1]) 
                den = (bins[m] - bins[m-1])
                fbank[m-1, k] = num/den
            for k in range(f_m_middle, f_m_right):
                num = (bins[m+1] - k) 
                den = bins[m+1] - bins[m]
                fbank[m-1, k] = num / den 

        temp_mfcc_data = []
        for s in sounds:
            subset = data[s.start: s.stop]
            # max_j = math.floor((len(subset)-N)/K)
            frame_data = chunk_data.chunk_data(subset, int(N), int(N-K))

            frame_energy = np.diag(np.dot(frame_data, frame_data.T))
            # print(frame_data.shape)
            apply_window = True
            if apply_window :
                window = np.hamming(len(frame_data[0]))
                for i in range(frame_data.shape[0]):
                    frame_data[i] = frame_data[i]*window

            data_fft = np.absolute(np.fft.rfft(frame_data, N))
            pow_frames = ((1.0/N) * ((data_fft) ** 2))

            filter_banks = np.dot(pow_frames, fbank.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
            filter_banks = 20*np.log10(filter_banks)  # dB
            num_ceps = 12
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

            (nframes, ncoeff) = mfcc.shape
            n = np.arange(ncoeff)
            cep_lifter = 22
            lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
            mfcc *= lift  #*

        #     for i in range(mfcc.shape[0]):
        #         samples = mfcc[i]
        #         # training_data = np.vstack((training_data, samples))
        #         if 'TRAIN' in f :
        #             train_sentence_data.append(samples)
        #             train_sentence_sounds.append(s.phoneme)
        #         else :
        #             test_sentence_data.append(samples)
        #             test_sentence_sounds.append(s.phoneme)

        # if 'TRAIN' in f :
        #     training_data.append(train_sentence_data)
        #     training_sounds.append(train_sentence_sounds)
        # else:
        #     test_data.append(test_sentence_data)
        #     test_sounds.append(test_sentence_sounds)

            for i in range(mfcc.shape[0]):
                samples = np.append(mfcc[i], frame_energy[i])
                temp_mfcc_data.append(samples)
                if 'TRAIN' in f :
                    train_sentence_sounds.append(s.phoneme)
                else :
                    test_sentence_sounds.append(s.phoneme)

        temp_fderiv_data = []
        sample_count = len(temp_mfcc_data)
        feature_count = len(temp_mfcc_data[0])
        for i in range(sample_count):
            if i == 0 :
                prev_sample = np.zeros((feature_count))
            else:
                prev_sample = temp_mfcc_data[i-1]

            if i == sample_count - 1 :
                next_sample = np.zeros((feature_count))
            else:
                next_sample = temp_mfcc_data[i+1]

            first_deriv = (next_sample - prev_sample) / 2.0

            temp_fderiv_data.append(first_deriv)
            # print()

        temp_sderiv_data = []
        for i in range(sample_count):
            if i == 0 :
                prev_sample = np.zeros((feature_count))
            else:
                prev_sample = temp_fderiv_data[i-1]

            if i == sample_count - 1 :
                next_sample = np.zeros((feature_count))
            else:
                next_sample = temp_fderiv_data[i+1]

            second_deriv = (next_sample - prev_sample) / 2.0

            temp_sderiv_data.append(second_deriv)
            # print()

            # temp_sdata = np.append(temp_mfcc_data,temp_fderiv_data, axis=1)
            # temp_sdata = np.append(temp_sdata, temp_sderiv_data, axis=1)
            # sentence_data.append(temp_sdata)
            
        temp_data = np.append(temp_mfcc_data, temp_fderiv_data, axis=1)
        temp_data = np.append(temp_data, temp_sderiv_data, axis=1)
        if 'TRAIN' in f :
            train_sentence_data = temp_data
        else :
            test_sentence_data = temp_data
            


        if 'TRAIN' in f :
            training_data.append(train_sentence_data)
            training_sounds.append(train_sentence_sounds)
        else:
            test_data.append(test_sentence_data)
            test_sounds.append(test_sentence_sounds)

        

        # save_region_data(region_to_process, training_data)

        


tr_d = combine_data(training_data, training_sounds)
te_d = combine_data(test_data, test_sounds)

save_data(f"mfcc39_sentences_features_{region_to_process}.json",  tr_d=tr_d, te_d=te_d)


print('end data')