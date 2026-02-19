import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

# extract pkl data
file_path = sys.argv[1]

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

# declare the axis labels
second_axis_label = np.arccos(2*np.linspace(-0.5, 0.5, azim_fft)) / np.pi*180
third_axis_label = np.linspace(0, 1, range_fft) * max_range
X,Y = np.meshgrid(third_axis_label, second_axis_label)

# plot the visualization
fig = plt.figure()
ax = plt.axes(projection = '3d')

for i in range(second_fft.shape[0]):
    ax.clear()
    ax.plot_surface(X, Y, abs(second_fft[i]))
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    ax.set_zlabel('Magnitude')
    plt.pause(0.1)

plt.show()
