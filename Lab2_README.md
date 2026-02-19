# Instructions to Run lab2_wireless_code.py

# Step 1
Run 'pip install -r requirements.txt'

# Step 2
Run 'python lab2_wireless_code.py [inputfilename]' to run TASK1 and TASK2 on an individual file

Make sure to replace 'inputfilename' with the path to the input pkl file to be read

# Step 3
After running, you should get two png files (for example:
                                            '113.pkl_pixelwise.png'
                                            '113.pkl_temporal.png'
                and the compressed file (for example):
                                            '113_compressed.pkl'
As is mentioned in the code, I applied thresholding and data type reduction in the compression.
This resulted in somewhere to 2-3x smaller a file. Due to this being lossy, I have also calculated the error.

# Step 4
All necessary data will be output, and attached are the 'requirements.txt' file and a Word file with my
received pngs of the histograms.