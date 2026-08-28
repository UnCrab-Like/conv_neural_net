# Source - https://stackoverflow.com/a
# Posted by Punnerud, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-23, License - CC BY-SA 4.0
from PIL import Image
import gzip
import numpy as np
import cupy as cp
#import matplotlib.pyplot as plt

# f = gzip.open("FCT/Python/train-images-idx3-ubyte.gz", "r")
f = gzip.open("train-images-idx3-ubyte.gz", "r")


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
image2.show()

# f = gzip.open("FCT/Python/train-labels-idx1-ubyte.gz", "r")
f = gzip.open("train-labels-idx1-ubyte.gz", "r")

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


def deriv_max_pooling(input):
    input = cp.asarray(input)
    batch, height, width = input.shape
    input_reshaped = input.reshape((batch, height // 2, 2, width // 2, 2))
    #input_reshaped = input.reshape((batch, height // 2, 2, width))
    #derived_max = [[cp.where(input_reshaped[:, i, : j, :] == input_reshaped[:, i, :, j, :].max(), 1, 0) for j in range(height//2)]for i in range(width//2)]
    #return input_reshaped[:, 1, :, 0:4]
    derived_max = [cp.where(input_reshaped[:, i, :, j, :] == input_reshaped[:, i, :, j, :].max(), 1, 0) for i in range(height//2) for j in range(width//2)]
    derived_max = cp.stack(derived_max, axis=2).reshape(batch, height, width)
    return derived_max

def deriv_conv_layer(input, filter):
    filter = cp.asarray(filter)
    input = cp.asarray(input)
    a = cp.zeros((input.shape[0], input.shape[1], input.shape[2]))

    #a[:, 0:3, 0:3] += filter

    for i in range(input.shape[1]):
        for j in range(input.shape[2]):
            a[:, i, j] = cp.sum(filter * input[:, i : i + filter.shape[1], j : j + filter.shape[2]])

    return a


        

    




array = cp.array([[[0, 0, 2, 1],
                   [0, 2, 3, 0],
                   [2, 3, 2, 0],
                   [1, 2, 0, 0]]])

filter = cp.array([[[1, 2, 1],
                    [2, 3, 1],
                    [2, 2, 1]]])

error = cp.array([[[1, 1], [1, 1]]])



#print(deriv_max_pooling(array))
print(deriv_conv_layer(array, filter))

