# Source - https://stackoverflow.com/a
# Posted by Punnerud, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-23, License - CC BY-SA 4.0
from PIL import Image
import gzip
import numpy as np
import matplotlib.pyplot as plt

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
plt.imshow(image)
plt.show()

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

print(labels)

