import os
import random

import numpy as np
from PIL import Image
from keras import optimizers
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, Concatenate
from keras.models import Model
from tqdm import tqdm
from keras import backend as K
from utils import get_image_file_paths

img_width, img_height = 224, 224


def preprocess_input(np_img):
    """tf-style preprocessing: inputs between -1 and 1"""
    np_img = np_img / 127.5
    np_img = np_img - 1.
    return np_img


def generate_examples(num_examples):
    x1_data = []
    x2_data = []
    y_data = []

    for i in tqdm(range(num_examples), desc="Preparing data"):
        # Pick two images at random
        image1_file_path, image2_file_path = random.sample(image_file_paths, 2)

        # Load the images and convert them to numpy arrays
        img1 = np.array(Image.open(image1_file_path).convert("RGB"))
        img1 = preprocess_input(img1)
        x1_data.append(img1)
        img2 = np.array(Image.open(image2_file_path).convert("RGB"))
        img2 = preprocess_input(img2)
        x2_data.append(img2)

        # Append target vector
        is_2nd_wavelength_longer = image2_file_path > image1_file_path
        output_vector = [0, 1] if is_2nd_wavelength_longer else [1, 0]
        y_data.append(output_vector)

    x1_data = np.array(x1_data)
    x2_data = np.array(x2_data)
    y_data = np.array(y_data)
    return [x1_data, x2_data], y_data


if __name__ == "__main__":
    random.seed(42)

    image_file_paths = get_image_file_paths()

    submodel_inputs = []
    submodel_outputs = []
    num_submodels = 2
    for i in range(num_submodels):
        submodel = MobileNet(
            alpha=0.25,
            weights="imagenet",
            include_top=False,
            input_shape=(img_width, img_height, 3),
        )
        for layer in submodel.layers:
            # Make layer name unique
            layer.name = "submodel{}_{}".format(i, layer.name)
            layer.trainable = True

        x = submodel.output
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)
        submodel_inputs.append(submodel.input)
        submodel_outputs.append(x)

    merged_output = Concatenate()(submodel_outputs)
    merged_output = Dense(32, activation="relu")(merged_output)
    merged_output = Dense(2, activation="softmax")(merged_output)

    # Create the final model
    final_model = Model(inputs=submodel_inputs, outputs=merged_output)

    # print(final_model.summary())

    # Compile the model
    final_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )

    num_examples_per_epoch = 512
    num_epochs = 50

    # Train the model
    for i in range(num_epochs):
        x_data, y_data = generate_examples(num_examples_per_epoch)
        new_lr = 0.001 / (1 + i)
        print('Learning rate: {0:.6f}'.format(new_lr))
        K.set_value(final_model.optimizer.lr, new_lr)
        final_model.fit(
            x_data,
            y_data,
            batch_size=16,
            shuffle=False,
        )

    final_model.save(os.path.join(os.path.dirname(__file__), "models", "final.h5"))
