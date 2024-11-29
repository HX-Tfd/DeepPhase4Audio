"""
Based on https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/Network.py     
"""
#Start Parameter Section
window = 2.0 #time duration of the time window
fps = 60 #fps of the motion capture data
joints = 26 #joints of the character skeleton

frames = int(window * fps) + 1
input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-component of each joint)
phase_channels = 5 #desired number of latent phase channels (usually between 2-10)

epochs = 10
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-4
restart_period = 10
restart_mult = 2

plotting_interval = 500 #update visualization at every n-th batch (visualization only)
pca_sequence_count = 100 #number of motion sequences visualized in the PCA (visualization only)
test_sequence_ratio = 0.01 #ratio of randomly selected test sequences (visualization only)
#End Parameter Section