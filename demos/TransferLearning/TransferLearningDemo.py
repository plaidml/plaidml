import warnings
warnings.simplefilter('ignore')

import argparse

import ngraph_bridge
ngraph_bridge.enable()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from IPython.display import display

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import optimizers

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

import ipywidgets as widgets

from IPython.display import clear_output


class ProgressBar(tf.keras.callbacks.Callback):

    def __init__(self, demo):
        self.demo = demo
        self.epoch = 0
        self.train_value = 0
        self.train_steps_per_epoch = demo.train_generator.n // demo.batch_size

    def on_train_begin(self, logs=None):
        self.demo.progress_bar.value = self.demo.progress_bar.min
        self.demo.progress_bar.max = self.train_steps_per_epoch * self.demo.epochs
        self.demo.progress_text.value = "Training"
        self.train_value = 0

    def on_train_end(self, logs=None):
        self.demo.progress_bar.value = self.demo.progress_bar.max
        self.demo.progress_text.value = "Training Complete"

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.demo.progress_text.value = "Training epoch " + str(epoch + 1)

    def on_epoch_end(self, epoch, logs=None):
        self.demo.epoch_text.value = "Epoch " + str(epoch + 1) + " val_acc: " + str(
            logs.get('val_acc'))

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        self.demo.progress_bar.value = self.epoch * self.train_steps_per_epoch + batch
        if logs:
            self.demo.progress_text.value = 'Training epoch ' + str(self.epoch +
                                                                    1) + ' loss: ' + str(
                                                                        logs.get('loss'))

    def on_test_begin(self, logs=None):
        self.demo.progress_text.value = "Validating epoch " + str(self.epoch + 1)

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        self.demo.progress_text.value = "Classifying"

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.demo.progress_bar.value = batch

    def on_predict_end(self, batch, logs=None):
        self.demo.progress_bar.value = self.demo.progress_bar.max
        self.demo.progress_text.value = "Classification Complete"


class Demo:
    # Cats & Dogs classes
    NUM_CLASSES = 2

    # RGB
    CHANNELS = 3

    RESNET50_POOLING_AVERAGE = 'avg'
    DENSE_LAYER_ACTIVATION = 'softmax'
    OBJECTIVE_FUNCTION = 'categorical_crossentropy'

    # Common accuracy metric for all outputs, but can use different metrics for different output
    LOSS_METRICS = ['accuracy']

    IMAGE_SIZE = 160  # All images will be resized to 160x160
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    batch_size = 32
    epochs = 2
    fine_tune_at = 100

    ngraph_backends = ngraph_bridge.list_backends()
    ngraph_backends.append('DISABLED')
    ngraph_bridge.set_backend(ngraph_backends[0])

    base_model = None
    model = None

    # GUI Elements
    out = widgets.Output(layout={'border': '1px solid black'})
    out_initial = widgets.Output(layout={'border': '1px solid black'})
    out_stats = widgets.Output(layout={'border': '1px solid black'})

    model_dropdown = widgets.Dropdown(options=['ResNet50', 'MobileNet v2'],
                                      value='ResNet50',
                                      description='Model:')
    ngraph_dropdown = widgets.Dropdown(options=ngraph_backends,
                                       value=ngraph_backends[0],
                                       description='nGraph:')

    progress_bar = widgets.IntProgress()
    progress_text = widgets.Label()
    epoch_text = widgets.Label()
    progress_box = widgets.HBox([progress_bar, progress_text, epoch_text])

    batch_slider = widgets.IntSlider(min=1, max=100, value=batch_size, description='Batch Size:')
    epoch_slider = widgets.IntSlider(min=1, max=16, value=epochs, description='Epochs:')
    fine_tune_slider = widgets.IntSlider(min=1,
                                         max=500,
                                         value=fine_tune_at,
                                         description='Fine Tune at Layer:')

    train_button = widgets.Button(description='Train', disabled=True)
    classify_button = widgets.Button(description='Classify', disabled=True)
    fine_tune_button = widgets.Button(description='Fine Tune', disable=True)
    shuffle_button = widgets.Button(description='Shuffle')

    training_tab = widgets.VBox(children=[
        model_dropdown, ngraph_dropdown, batch_slider, epoch_slider, train_button, classify_button,
        shuffle_button
    ])
    fine_tuning_tab = widgets.VBox(children=[
        batch_slider, epoch_slider, fine_tune_slider, fine_tune_button, classify_button,
        shuffle_button
    ])
    tab = widgets.Tab(children=[training_tab, fine_tuning_tab])
    tab.set_title(0, 'Training')
    tab.set_title(1, 'Fine Tuning')
    gui = widgets.VBox(children=[tab, progress_box, out_initial, out])

    def __init__(self, gui=1, training=0, model='ResNet50', verbose=1):
        self.gui = gui
        self.verbose = verbose

        if gui:
            self.gui = widgets.VBox(
                children=[self.tab, self.progress_box, self.out_initial, self.out, self.out_stats])
            display(self.init_gui())

        # Images
        self.test_class_indices = []
        self.test_dir = None
        self.init_images()

        self.sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.predicted_class_indices_init = []
        self.wrong_guesses = []

        self.train_button.disabled = False
        self.fine_tune_button.disabled = False

        self.init_model()

        if training:
            self.train_model(self.train_button, model=model)

    def init_images(self):
        zip_file = tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
            fname="cats_and_dogs_filtered.zip",
            extract=True)
        base_dir, _ = os.path.splitext(zip_file)
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')
        self.test_dir = os.path.join(base_dir, 'test')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # Directory with our training cat pictures
        train_cats_dir = os.path.join(train_dir, 'cats')

        # Directory with our training dog pictures
        train_dogs_dir = os.path.join(train_dir, 'dogs')

        # Directory with our validation cat pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')

        # Directory with our validation dog pictures
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')

        # Directory with our test cat pictures
        test_cats_dir = os.path.join(self.test_dir, 'cats')
        if not os.path.exists(test_cats_dir):
            os.makedirs(test_cats_dir)
            for i in range(900, 1000):
                os.rename(train_cats_dir + '/cat.' + str(i) + '.jpg',
                          test_cats_dir + '/cat.' + str(i) + '.jpg')

        # Directory with our test dog pictures
        test_dogs_dir = os.path.join(self.test_dir, 'dogs')
        if not os.path.exists(test_dogs_dir):
            os.makedirs(test_dogs_dir)
            for i in range(900, 1000):
                os.rename(train_dogs_dir + '/dog.' + str(i) + '.jpg',
                          test_dogs_dir + '/dog.' + str(i) + '.jpg')

        # Preprocess images
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        with self.out_stats:
            # Flow training images in batches of 20 using train_datagen generator
            self.train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                batch_size=self.batch_size,
                class_mode='categorical')

            # Flow validation images in batches of 20 using test_datagen generator
            self.validation_generator = validation_datagen.flow_from_directory(
                validation_dir,  # Source directory for the validation images
                target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                batch_size=self.batch_size,
                class_mode='categorical')

            # Flow validation images in batches of 20 using test_datagen generator
            self.test_generator = validation_datagen.flow_from_directory(
                self.test_dir,  # Source directory for the test images
                target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                batch_size=self.batch_size,
                class_mode=None,
                shuffle=False,
                seed=42)

        # Test Correct Values (0 Cat, 1 Dog)
        for file in self.test_generator.filenames:
            if "cat" in file:
                self.test_class_indices.append(0)
            elif "dog" in file:
                self.test_class_indices.append(1)
            else:
                print("Error, unclassifiable image " + file)

    def init_model(self, b=None):
        self.out_initial.clear_output()
        self.out.clear_output()
        self.out_stats.clear_output()

        # Initial prediction
        self.progress_bar.max = 13
        self.progress_text.value = "Calculating Initital Predictions"

        with self.out:
            self.compile_model(self.model_dropdown.value)
        predictions_initial = self.model.predict_generator(self.test_generator,
                                                           verbose=0,
                                                           callbacks=[ProgressBar(self)])
        self.predicted_class_indices_init = np.argmax(predictions_initial, axis=1)

        # ~ operator is equivalent to -x-1; so 0 becomes -1, 1 becomes -2; add 2 to implement xnor
        guesses = ~(self.predicted_class_indices_init ^ self.test_class_indices) + 2

        # Randomly select 9 images that guessed incorrectly
        self.wrong_guesses = []
        while len(self.wrong_guesses) < 9:
            randomIndex = random.randint(0, len(self.predicted_class_indices_init) - 1)
            if not guesses[randomIndex]:
                self.wrong_guesses.append(randomIndex)

        if self.gui:
            with self.out_initial:
                f, ax = plt.subplots(3, 3, figsize=(15, 15))

                for i in range(0, 9):
                    test_image = os.path.join(self.test_dir,
                                              self.test_generator.filenames[self.wrong_guesses[i]])
                    imgRGB = mpimg.imread(test_image)

                    predicted_class = "Dog" if self.predicted_class_indices_init[
                        self.wrong_guesses[i]] else "Cat"

                    ax[i // 3, i % 3].imshow(imgRGB)
                    ax[i // 3, i % 3].axis('off')
                    ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='r')

                    if predicted_class.lower() in self.test_generator.filenames[self.wrong_guesses[i]]:
                        ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='g')
                display(plt)

    def on_model_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.classify_button.disabled = True

            if change['new'] == 'ResNet50':
                self.epoch_slider.min = 1
                self.epoch_slider.max = 16
                self.epoch_slider.value = 2
            if change['new'] == 'MobileNet v2':
                self.epoch_slider.min = 2
                self.epoch_slider.max = 50
                self.epoch_slider.value = 10

    def on_ngraph_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            i = self.ngraph_backends.index(change['new'])

            if self.ngraph_backends[i] == 'DISABLED':
                self.use_ngraph = False
                ngraph_bridge.disable()
            else:
                self.use_ngraph = True
                ngraph_bridge.enable()
                ngraph_bridge.set_backend(self.ngraph_backends[i])

    def init_gui(self):
        self.model_dropdown.observe(self.on_model_change)
        self.ngraph_dropdown.observe(self.on_ngraph_change)
        self.train_button.on_click(self.train_model)
        self.shuffle_button.on_click(self.init_model)
        self.classify_button.on_click(self.classify)
        self.fine_tune_button.on_click(self.train_model)

        return self.gui

    def train_model(self, b=None, epochs=1, batch_size=32, model='ResNet50', fine_tune_at=0):
        if self.gui:
            self.train_button.disabled = True
            self.epochs = self.epoch_slider.value
            self.batch_size = self.batch_slider.value
            model = self.model_dropdown.value
            self.progress_text.value = ''
        else:
            self.epochs = epochs
            self.batch_size = batch_size

        self.out.clear_output()

        if b == self.fine_tune_button:
            fine_tune_at = self.fine_tune_slider.value
        else:
            fine_tune_at = 0

        self.compile_model(model, fine=fine_tune_at)
        steps_per_epoch = self.train_generator.n // self.batch_size
        validation_steps = self.validation_generator.n // self.batch_size

        with self.out_stats:
            history = self.model.fit_generator(self.train_generator,
                                               steps_per_epoch=steps_per_epoch,
                                               epochs=self.epochs,
                                               workers=8,
                                               validation_data=self.validation_generator,
                                               validation_steps=validation_steps,
                                               verbose=0,
                                               callbacks=[ProgressBar(self)])
        self.classify_button.disabled = False
        # Test model immediately after training
        self.classify(b)

        self.train_button.disabled = False

        return history

    def compile_model(self, modelName, fine=0):
        if modelName == 'ResNet50':
            self.base_model = ResNet50(input_shape=self.IMAGE_SHAPE,
                                       include_top=False,
                                       weights='imagenet')
        elif modelName == 'MobileNet v2':
            self.base_model = MobileNetV2(input_shape=self.IMAGE_SHAPE,
                                          include_top=False,
                                          weights='imagenet')

        # GUI element update
        self.fine_tune_slider.max = len(self.base_model.layers)

        with self.out_stats:
            print('Setting base model to ' + modelName)

        # Fine Tuning
        if fine:
            self.base_model.trainable = True
            # Fine tune from this layer onwards
            self.fine_tune_at = fine

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_model.layers[:self.fine_tune_at]:
                layer.trainable = False

            with self.out_stats:
                print('Fine tuning at layer ' + str(fine))

        # Training
        else:
            self.base_model.trainable = False

        # Add layers
        self.model = tf.keras.Sequential([
            self.base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(self.NUM_CLASSES, activation=self.DENSE_LAYER_ACTIVATION)
        ])
        self.model.compile(optimizer=self.sgd,
                           loss=self.OBJECTIVE_FUNCTION,
                           metrics=self.LOSS_METRICS)

    def classify(self, b):
        if b and b.description == 'Classify':
            out.clear_output()

        with self.out_stats:
            probabilities = self.model.predict_generator(self.test_generator,
                                                         verbose=0,
                                                         callbacks=[ProgressBar(self)])

        predicted_class_indices = np.argmax(probabilities, axis=1)

        if self.gui:
            with self.out:
                f, ax = plt.subplots(3, 3, figsize=(15, 15))

                for i in range(0, 9):
                    test_image = os.path.join(self.test_dir,
                                              self.test_generator.filenames[self.wrong_guesses[i]])
                    imgRGB = mpimg.imread(test_image)

                    predicted_class = "Dog" if predicted_class_indices[
                        self.wrong_guesses[i]] else "Cat"

                    ax[i // 3, i % 3].imshow(imgRGB)
                    ax[i // 3, i % 3].axis('off')
                    ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='r')

                    if predicted_class.lower() in self.test_generator.filenames[self.wrong_guesses[i]]:
                        ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='g')
                display(f)

        if self.verbose:
            with self.out_stats:
                wrong = ~(predicted_class_indices ^ self.test_class_indices) + 2
                print("Correct Matrix")
                print(wrong)

            print("Total guessed:", wrong.shape[0])
            print("Accuracy:", np.count_nonzero(wrong)/wrong.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TransferLearningDemo')
    parser.add_argument('--gui', help='shows the GUI of the demo', action='store_true')
    parser.add_argument('--training', help='performs the training phase of the demo', action='store_true')
    parser.add_argument('--network_type', help='selects the network used for training/classification [ResNet50]/MobileNet V2')
    parser.add_argument('--quiet', help='disables most logging', action='store_false')
    args = parser.parse_args()
    nw = 'ResNet50'
    if args.network_type == 'ResNet50' or args.network_type == 'MobileNet V2':
        nw = args.network_type
    Demo(args.gui, args.training, nw, args.quiet)
