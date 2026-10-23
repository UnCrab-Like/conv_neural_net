# Source - https://stackoverflow.com/a
# Posted by Punnerud, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-23, License - CC BY-SA 4.0a
from PIL import Image
import gzip
import numpy as np
import cupy as cp
from flask import Flask
#import matplotlib.pyplot as plt

# f = gzip.open("FCT/Python/train-images-idx3-ubyte.gz", "r")
f = gzip.open("training_data/train-images-idx3-ubyte.gz", "r")


image_size = 28
num_images = 5

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

# Source - https://stackoverflow.com/a
# Posted by Punnerud, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-23, License - CC BY-SA 4.0

image = np.asarray(data[2]).squeeze()
#plt.imshow(image)
#plt.show()

# image2 = Image.frombuffer("RGB", (image_size, image_size), data[2])
image2 = Image.fromarray(image)
image2 = image2.resize((129, 129), Image.Resampling.BICUBIC)
# image2.show()

# f = gzip.open("FCT/Python/train-labels-idx1-ubyte.gz", "r")
f = gzip.open("training_data/train-labels-idx1-ubyte.gz", "r")

f.read(8)
"""
for i in range(0, 50):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)
"""
buf = f.read(2)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

# print(labels)


def deriv_max_pooling_rework(input):
    input = cp.asarray(input)
    batch, filter_batch, height, width = input.shape
    return_height, return_width = height // 2, width // 2

    # 1. Expand into 6D grid tensor
    input_reshaped = input.reshape((batch, filter_batch, return_height, 2, return_width, 2))
    
    # 2. Get the max inside each 2x2 block instantly on the GPU (axes 3 and 5)
    block_maxes = input_reshaped.max(axis=(3, 5), keepdims=True)
    
    # 3. Create the 1s and 0s mask instantly across all batches and channels
    derived_max = cp.where(input_reshaped == block_maxes, 1, 0)
    
    # 4. Transpose and reshape back to your 4D layout
    derived_max = derived_max.transpose(0, 1, 2, 3, 4, 5)
    return derived_max.reshape(batch, filter_batch, height, width)


def deriv_conv_layer(input, filter):
    filter = cp.asarray(filter)
    input = cp.asarray(input)
    a = cp.zeros((input.shape[0], input.shape[1], input.shape[2]))

    #a[:, 0:3, 0:3] += filter

    for i in range(input.shape[1]):
        for j in range(input.shape[2]):
            a[:, i, j] = cp.sum(filter * input[:, i : i + filter.shape[1], j : j + filter.shape[2]])

    return a



def filter_image(img_matrix, filter):
    ### Image initialisation
    
    ### Dimensions of the image.
    batch, height, width = img_matrix.shape
    filter = cp.asarray(filter)
    f_batch, f_height, f_width = filter.shape
    
    output_height = height - f_height + 1 # 3 being filter height
    output_width = width - f_width + 1 # 3 being filter height

    ### Grid initialisation.
    fil_row = cp.repeat(cp.arange(f_height), f_height)  # 9x
    fil_col = cp.tile(cp.arange(f_width), f_width)  # 9x

    matrix_row = cp.repeat(cp.arange(output_height), output_width)  # nx
    matrix_column = cp.tile(cp.arange(output_width), output_height )  # mx

    cords_x = fil_row[:, cp.newaxis] + matrix_row  # 9xn
    cords_y = fil_col[:, cp.newaxis] + matrix_column  # 9xm

    cords_x = cp.asarray(cords_x).astype(cp.int32)
    cords_y = cp.asarray(cords_y).astype(cp.int32)

    img_matrix_ = cp.asarray(img_matrix)

    cord_matrix = img_matrix_[:, cords_x, cords_y]

    ### Long process giving the correct value.
    return (cp.asarray([cp.sum(filter[i].ravel()[cp.newaxis, cp.newaxis, :, cp.newaxis] * cord_matrix, axis=2) for i in range(f_batch)])).reshape(batch, f_batch, output_height, output_width)
    

def max_pooling(img_matrix, mode='L'):
    batch, filter, height, width = img_matrix.shape

    pooled = img_matrix.reshape(
        batch, filter, height // 2,
        2,
        width // 2,
        2,
    ).max(axis=(3, 5))
    return pooled

# Notes for future self: filter can be unraveled into a [9*a] where a is the filter amount (Parallel filtering.) From there, you keep it in its size. [H*a , L]



array = cp.array([[[[0 , 0 , 2 , 1 ],
                   [0 , 2 , 3 , 0 ],
                   [2 , 3 , 2 , 0 ],
                   [1 , 2 , 0 , 0 ]],
                   
                  [[0 , 1 , 2 , 3 ],
                   [4 , 5 , 6 , 7 ],
                   [8 , 9 , 10, 11],
                   [12, 13, 14, 15]]
                   ]])

# array = cp.array([[[1 , 2 , 3 , 4 ],
#                    [5 , 6 , 7 , 8 ],
#                    [9 , 10, 11, 12],
#                    [13, 14, 15, 16]]])

# filter = cp.array([[[1, 2, 1],
#                     [2, 3, 1],
#                     [2, 2, 1]],

#                    [[3, 2, 0],
#                     [1, 1, 0],
#                     [1, 0, 0]]])

filter = cp.array([[[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]],

                   [[2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2]]])

# filter = cp.array([[[1, 2, 1],
#                     [2, 3, 1],
#                     [2, 2, 1]]])

error = cp.array([[[1, 1], [1, 1]]])



print(filter_image(array, filter))

# fil = filter_image(array, filter)

# print(fil)
# print(max_pooling(fil))

