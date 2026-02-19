import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# TASK 1
def analyze_sparsity(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    frames = data['frames']

    # declare necessary variables
    range_fft = 512
    azim_fft = 64

    # run the two FFTs and shift accordingly
    first_fft = np.fft.fft(frames, n = range_fft, axis = 2)
    second_fft = np.fft.fft(first_fft, n = azim_fft, axis = 1)
    second_fft = np.fft.fftshift(second_fft, axes = 1)

    # pixel differences
    magnitudes = np.abs(second_fft).flatten()

    # plot visualization
    plt.figure(figsize = (10, 6))
    plt.hist(magnitudes, bins = 100, log = True, color = 'red', alpha = 0.7)
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    plt.title(f'Pixel-Wise Dist - {os.path.basename(file_path)}')
    plt.grid(True, alpha = 0.3)
    # plt.savefig(f'{os.path.basename(file_path)}_pixelwise.png')
    # Above no longer needed since the files are being submitted in a Word doc

    # temporal sparsity
    frame_diffs = np.diff(second_fft, axis = 0)
    diff_magnitudes = np.abs(frame_diffs).flatten()

    # plot visualization
    plt.figure(figsize = (10, 6))
    plt.hist(diff_magnitudes, bins = 100, log = True, color = 'red', alpha = 0.7)
    plt.xlabel('Magnitude of Frames')
    plt.ylabel('Count')
    plt.title(f'Temporal Sparsity - {os.path.basename(file_path)}')
    plt.grid(True, alpha = 0.3)
    # plt.savefig(f'{os.path.basename(file_path)}_temporal.png')
    # Above no longer needed since the files are being submitted in a Word doc

    # calculate entropies for pixel
    hist1, bin_edges1 = np.histogram(magnitudes, bins = 100, density = True)
    bin_width1 = bin_edges1[1] - bin_edges1[0]
    probs1 = hist1 * bin_width1
    probs1 = probs1[probs1 > 0]
    entropy_pixelwise = -np.sum(probs1 * np.log2(probs1))

    # calculate entropies for temporal
    hist2, bin_edges2 = np.histogram(diff_magnitudes, bins = 100, density = True)
    bin_width2 = bin_edges2[1] - bin_edges2[0]
    probs2 = hist2 * bin_width2
    probs2 = probs2[probs2 > 0]
    entropy_temporal = -np.sum(probs2 * np.log2(probs2))

    print(f"Pixel-wise Entropy: {entropy_pixelwise:.4f} bits")
    print(f"Temporal Entropy: {entropy_temporal:.4f} bits")

    return entropy_pixelwise, entropy_temporal


# TASK 2
def compress_file(file_path, output_path, percentile=90):

    # I compress the data into a complex64, and also apply thresholding
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    frames = data['frames']

    # calculate and apply threshold, while also calculating sparsity
    threshold = np.percentile(np.abs(frames), percentile)
    frames_thresholded = np.where(np.abs(frames) < threshold, 0, frames)
    sparsity = np.sum(frames_thresholded == 0) / frames_thresholded.size * 100

    # convert to smaller data type and save
    frames_compressed = frames_thresholded.astype(np.complex64)
    data_compressed = {'frames': frames_compressed}
    with open(output_path, 'wb') as f:
        pickle.dump(data_compressed, f)

    # calculate compression
    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size

    # calculate error since we could be lossy
    mse = np.mean(np.abs(frames - frames_compressed)**2)
    relative_error = np.sqrt(mse) / np.mean(np.abs(frames)) * 100

    print(f"Original file size: {original_size / 1e6:.2f} MB")
    print(f"Compressed file size: {compressed_size / 1e6:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Relative Error: {relative_error:.4f}%")

    return {
        'file': os.path.basename(file_path),
        'original_size_MB': original_size / 1e6,
        'compressed_size_MB': compressed_size / 1e6,
        'compression_ratio': compression_ratio,
        'sparsity_percent': sparsity,
        'mse': mse,
        'relative_error_percent': relative_error
    }


# MAIN
if __name__ == "__main__":
        file_path = sys.argv[1]

        # Task 1
        analyze_sparsity(file_path)

        # Task 2
        output_path = file_path.replace('.pkl', '_compressed.pkl')
        compress_file(file_path, output_path, percentile=90)