from PIL import Image
import gzip
import os
#import numpy as np
import cupy as cp
import warnings
#import cv2


from src.front_prop import front_propagation
from src.back_prop import back_propagation
from src.adam_optimizer import AdamOptimizer

if __name__ == "__main__":
    f = gzip.open("training_data/train-images-idx3-ubyte.gz", "r")

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    training_data = cp.frombuffer(buf, dtype=cp.uint8).astype(cp.float32)
    training_data = training_data.reshape(num_images, image_size, image_size, 1)
    training_data = training_data / 255.0

    f = gzip.open("training_data/train-labels-idx1-ubyte.gz", "r")

    f.read(8)
    buf = f.read(num_images)
    training_labels = cp.frombuffer(buf, dtype=cp.uint8)

    permutation = cp.random.permutation(len(training_data))

    training_data = training_data[permutation, :, :]
    training_labels = training_labels[permutation]



    f = gzip.open("t10k-images-idx3-ubyte.gz", "r")

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    validation_data = cp.frombuffer(buf, dtype=cp.uint8).astype(cp.float32)
    validation_data = validation_data.reshape(num_images, image_size, image_size, 1)
    validation_data = validation_data / 255.0

    f = gzip.open("t10k-labels-idx1-ubyte.gz", "r")

    f.read(8)
    buf = f.read(num_images)
    validation_labels = cp.frombuffer(buf, dtype=cp.uint8)

    permutation = cp.random.permutation(len(validation_data))
    
    validation_data = validation_data[permutation, :, :]
    validation_labels = validation_labels[permutation]

    for i in range(10):
        