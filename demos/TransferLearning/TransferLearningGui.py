import os
import random
import subprocess

import ipywidgets as widgets
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output, display

import TransferLearningDemo


class ProgressBar(tf.keras.callbacks.Callback):

    def __init__(self, gui, demo):
        self.demo = demo
        self.gui = gui
        self.epoch = 0
        self.train_steps_per_epoch = self.demo.train_generator.n // self.demo.batch_size

    def on_train_begin(self, logs=None):
        self.total_epochs = self.gui.epoch_slider.value
        self.gui.progress_bar.value = self.gui.progress_bar.min
        self.gui.progress_bar.max = self.train_steps_per_epoch * self.total_epochs
        self.gui.epoch_text = ""

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.gui.progress_text.value = "Training epoch " + str(epoch + 1)
        pass

    def on_epoch_end(self, epoch, logs=None):
        #self.gui.epoch_text.value = "Epoch " + str(epoch + 1) + " val_acc: " + str(logs.get('val_acc'))
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        self.gui.progress_bar.value = self.epoch * self.train_steps_per_epoch + batch
        if logs:
            self.gui.progress_text.value = 'Training epoch ' + str(self.epoch +
                                                                   1) + ' loss: ' + str(
                                                                       logs.get('loss'))

    def on_test_begin(self, logs=None):
        self.gui.progress_text.value = "Validating epoch " + str(self.epoch + 1)
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        self.gui.progress_text.value = "Classifying"
        self.gui.progress_bar.max = self.demo.test_generator.n // self.demo.batch_size
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.gui.progress_bar.value = batch
        pass

    def on_predict_end(self, batch, logs=None):
        self.gui.progress_text.value = "Classification Complete"
        pass


class TrainingDemoGui():

    def __init__(self):
        self.ngraph_backends = ['PLAIDML', 'CPU', 'DISABLED']
        self.backend = self.ngraph_backends[0]
        self.demo = TransferLearningDemo.Demo(backend=self.backend)
        #self.progress_bar_callbacks = ProgressBar(self.demo)
        self.wrong_guess_indices = []

        # GUI Eleements
        self.initial = widgets.Output(layout={'border': '1px solid black'})
        self.updated = widgets.Output(layout={'border': '1px solid black'})
        self.log = widgets.Output(layout={'border': '1px solid black'})

        self.model_dropdown = widgets.Dropdown(options=['ResNet50', 'MobileNet v2'],
                                               value='ResNet50',
                                               description='Model:')
        self.ngraph_dropdown = widgets.Dropdown(options=self.ngraph_backends,
                                                value=self.ngraph_backends[0],
                                                description='nGraph:')
        self.batch_slider = widgets.IntSlider(min=1, max=100, value=16, description='Batch Size:')
        self.epoch_slider = widgets.IntSlider(min=1, max=16, value=5, description='Epochs:')
        self.train_button = widgets.Button(description='Train')
        self.training_tab = widgets.VBox(children=[
            self.model_dropdown, self.ngraph_dropdown, self.batch_slider, self.epoch_slider,
            self.train_button
        ])
        self.tab = widgets.Tab(children=[self.training_tab])
        self.tab.set_title(0, 'Training')
        self.progress_bar = widgets.IntProgress()
        self.progress_text = widgets.Label()
        self.epoch_text = widgets.Label()
        self.progress_box = widgets.HBox([self.progress_bar, self.progress_text, self.epoch_text])
        self.gui = widgets.VBox(children=[self.tab, self.progress_box, self.initial, self.updated])
        clear_output()
        display(self.gui)

        self.model_dropdown.observe(self.on_model_change)
        self.ngraph_dropdown.observe(self.on_ngraph_change)
        self.train_button.on_click(self.on_train_clicked)

    def on_model_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.train_button.disabled = False
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
            self.backend = self.ngraph_backends[i]

    def on_train_clicked(self, b=None):
        self.train_button.disabled = True
        # New instance of model
        with self.log:
            self.demo = TransferLearningDemo.Demo(epochs=self.epoch_slider.value,
                                                  batch_size=self.batch_slider.value,
                                                  model_name=self.model_dropdown.value,
                                                  backend=self.ngraph_dropdown.value,
                                                  verbose=0)
        self.progress_bar_callbacks = ProgressBar(self, self.demo)

        initial_pred = np.argmax(self.demo.predict([self.progress_bar_callbacks]), axis=1)

        # Randomly select 9 images that guessed incorrectly or try 1000 times
        i = 0
        while len(self.wrong_guess_indices) < 9 and i < 1000:
            i = i + 1
            randomIndex = random.randint(0, len(initial_pred) - 1)
            if initial_pred[randomIndex] != self.demo.test_class_indices[
                    randomIndex] and randomIndex not in self.wrong_guess_indices:
                self.wrong_guess_indices.append(randomIndex)

        # Display initial guesses
        with self.initial:
            print("Initial Predictions")
        self.display_images(initial_pred, self.initial)

        self.demo.train(self.epoch_slider.value, [self.progress_bar_callbacks])
        updated_pred = np.argmax(self.demo.predict([self.progress_bar_callbacks]), axis=1)

        # Display updated guesses
        with self.updated:
            print("Updated Predictions")
        self.display_images(updated_pred, self.updated)

        self.train_button.disabled = False

    def display_images(self, pred, window):
        # Display updated guesses
        f, ax = plt.subplots(3, 3, figsize=(15, 15))

        for i in range(0, 9):
            test_image = os.path.join(
                self.demo.test_dir,
                self.demo.test_generator.filenames[self.wrong_guess_indices[i]])
            imgRGB = mpimg.imread(test_image)
            predicted_class = "Dog" if pred[self.wrong_guess_indices[i]] else "Cat"

            ax[i // 3, i % 3].imshow(imgRGB)
            ax[i // 3, i % 3].axis('off')
            ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='r')

            if predicted_class.lower() in self.demo.test_generator.filenames[
                    self.wrong_guess_indices[i]]:
                ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='g')

        with window:
            plt.show()
