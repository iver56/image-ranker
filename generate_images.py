import math
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# Generate and store images
for i in tqdm(range(2000)):
    wavelength = 4 + 0.1 * i
    frequency = 1 / wavelength
    shape = (224, 224)
    img = np.zeros(shape, dtype=np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            img[y][x] = 0.5 + 0.5 * math.sin(math.pi * 2 * frequency * x)

    img = img * 255
    img_int = np.uint8(img)
    pil_image = Image.fromarray(img_int)
    pil_image.save(os.path.join("data", "{0:04d}.png".format(i)))
