#!/usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.mean(np.asarray(Image.open('img.jpg')), axis=2)
plt.imshow(img)
plt.show()
