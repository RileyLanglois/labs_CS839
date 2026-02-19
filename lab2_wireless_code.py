import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

# extract pkl data
file_path = '113.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

frames = data['frames']

# declare necessary variables
range_fft = 512
azim_fft = 64
max_range = 20

# run the two ffts and shift it accordingly
first_fft = np.fft.fft(frames, n = range_fft, axis = 2)
second_fft = np.fft.fft(first_fft, n = azim_fft, axis = 1)
second_fft = np.fft.fftshift(second_fft, axes = 1)

# print data for checking purposes
print(f"Data shape: {second_fft.shape}")
print(f"Number of frames: {second_fft.shape[0]}")
