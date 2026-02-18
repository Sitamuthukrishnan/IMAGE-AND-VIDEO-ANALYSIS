#code:
# Grayscale image output

import matplotlib.pyplot as plt
import numpy as np
from operator import add
from functools import reduce

# Upload image in Colab first
from google.colab import files
files.upload()

# Read image
img = plt.imread("img.jpg")

def split4(image):
    half_split = np.array_split(image, 2, axis=0)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

split_img = split4(img)

fig, axs = plt.subplots(2, 2, figsize=(6, 6))
axs[0, 0].imshow(split_img[0])
axs[0, 1].imshow(split_img[1])
axs[1, 0].imshow(split_img[2])
axs[1, 1].imshow(split_img[3])

for ax in axs.flat:
    ax.axis("off")

def concatenate4(nw, ne, sw, se):
    top = np.concatenate((nw, ne), axis=1)
    bottom = np.concatenate((sw, se), axis=1)
    return np.concatenate((top, bottom), axis=0)

full_img = concatenate4(
    split_img[0],
    split_img[1],
    split_img[2],
    split_img[3]
)

plt.figure(figsize=(4, 4))
plt.imshow(full_img)
plt.axis("off")
plt.show()

# Output:
<img width="410" height="520" alt="image" src="https://github.com/user-attachments/assets/55a4152b-8808-4b69-a2b7-aebe6cc64ee0" />
