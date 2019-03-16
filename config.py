import numpy as np

debug = True
image_source_dir = './dataset/facades/'
direction = 'b2a'
input_channel = 3  # input image channels
output_channel = 3  # output image channels
lr = 0.0002
epoch = 150
crop_from = 286
image_size = 256
batch_size = 20
combined_filepath = 'best_weights.h5'
generator_filepath = 'generator.h5'
seed = 9584
imagenet_mean = np.array([0.5, 0.5, 0.5])
imagenet_std = np.array([0.5, 0.5, 0.5])
