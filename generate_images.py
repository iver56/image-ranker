import os
import math

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

size = (224, 224)

fonts = {}


def get_font(font_size):
    if font_size not in fonts:
        fonts[font_size] = ImageFont.truetype("BebasNeue-Regular.ttf", font_size)
    return fonts[font_size]


# Generate and store images
for i in tqdm(range(1000)):

    image = Image.new("RGB", size, (255, 255, 255))

    # Drawing context
    ctx = ImageDraw.Draw(image)

    font_size = math.ceil(image.size[0] / 2)
    font = get_font(font_size)

    ctx.text(
        (size[0] / 2 - 0.6 * font_size, int(image.size[1] - 1.5 * font_size)),
        str(i),
        font=font,
        fill=(20, 20, 20, 20),
    )

    image.save(os.path.join("data", "{0:04d}.png".format(i)))
