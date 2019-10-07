import os
import random

import ipywidgets as widgets
import subprocess
import TransferLearningDemo

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython.display import display
from IPython.display import clear_output

ngraph_backends = ['PLAIDML', 'CPU', 'DISABLED']
batch_size = 16
epochs = 5
backend = ngraph_backends[0]


class ProgressBar(tf.keras.callbacks.Callback):

    def __init__(self, demo):
        self.epoch = 0
        self.train_steps_per_epoch = demo.train_generator.n // demo.batch_size

    def on_train_begin(self, logs=None):
        self.total_epochs = epoch_slider.value
        progress_bar.value = progress_bar.min
        progress_bar.max = self.train_steps_per_epoch * self.total_epochs
        epoch_text = ""
        #self.demo.progress_text.value = "Training"
        #self.train_value = 0

    def on_train_end(self, logs=None):
        #self.demo.progress_bar.value = self.demo.progress_bar.max
        #self.demo.progress_text.value = "Training Complete"
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        progress_text.value = "Training epoch " + str(epoch + 1)
        pass

    def on_epoch_end(self, epoch, logs=None):
        epoch_text.value = "Epoch " + str(epoch + 1) + " val_acc: " + str(logs.get('val_acc'))
        pass

    def on_train_batch_begin(self, batch, logs=None):
        #print("***" + str(self.epoch) + str(self.train_steps_per_epoch) + str(batch))
        pass

    def on_train_batch_end(self, batch, logs=None):
        progress_bar.value = self.epoch * self.train_steps_per_epoch + batch
        if logs:
            progress_text.value = 'Training epoch ' + str(self.epoch + 1) + ' loss: ' + str(
                logs.get('loss'))

    def on_test_begin(self, logs=None):
        progress_text.value = "Validating epoch " + str(self.epoch + 1)
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        progress_text.value = "Classifying"
        progress_bar.max = demo.test_generator.n // demo.batch_size
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        progress_bar.value = batch
        pass

    def on_predict_end(self, batch, logs=None):
        progress_text.value = "Classification Complete"
        pass


demo = TransferLearningDemo.Demo()
progress_info = ProgressBar(demo)


def on_model_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        classify_button.disabled = True

        if change['new'] == 'ResNet50':
            epoch_slider.min = 1
            epoch_slider.max = 16
            epoch_slider.value = 2
        if change['new'] == 'MobileNet v2':
            epoch_slider.min = 2
            epoch_slider.max = 50
            epoch_slider.value = 10


def on_ngraph_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        i = ngraph_backends.index(change['new'])
        backend = ngraph_backends[i]


def on_train_clicked(self, b=None):
    epochs = epoch_slider.value
    batch_size = batch_slider.value
    backend = ngraph_dropdown.value
    model = model_dropdown.value

    train_button.disabled = True

    # Everytime we train, get a new instance of the demo
    with out:
        demo = TransferLearningDemo.Demo(model=model,
                                         backend=backend,
                                         epochs=epochs,
                                         batch_size=batch_size,
                                         verbose=0)
    initial_prob = demo.predict([progress_info])
    initial_pred = np.argmax(initial_prob, axis=1)

    # Randomly select 9 images that guessed incorrectly
    wrong_guess_indices = []
    while len(wrong_guess_indices) < 9:
        randomIndex = random.randint(0, len(initial_pred) - 1)
        if initial_pred[randomIndex] != demo.test_class_indices[randomIndex]:
            wrong_guess_indices.append(randomIndex)

    # Display initial guesses
    f, ax = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(0, 9):
        test_image = os.path.join(demo.test_dir,
                                  demo.test_generator.filenames[wrong_guess_indices[i]])
        imgRGB = mpimg.imread(test_image)
        predicted_class = "Dog" if initial_pred[wrong_guess_indices[i]] else "Cat"

        ax[i // 3, i % 3].imshow(imgRGB)
        ax[i // 3, i % 3].axis('off')
        ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='r')

        if predicted_class.lower() in demo.test_generator.filenames[wrong_guess_indices[i]]:
            ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='g')
    print('Initial Prediction:')
    display(plt)

    demo.train(model, backend, epochs, batch_size, [progress_info])

    # Update prediction with trained model
    updated_prob = demo.predict([progress_info])
    updated_pred = np.argmax(updated_prob, axis=1)

    # Display updated guesses
    f, ax = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(0, 9):
        test_image = os.path.join(demo.test_dir,
                                  demo.test_generator.filenames[wrong_guess_indices[i]])
        imgRGB = mpimg.imread(test_image)
        predicted_class = "Dog" if updated_pred[wrong_guess_indices[i]] else "Cat"

        ax[i // 3, i % 3].imshow(imgRGB)
        ax[i // 3, i % 3].axis('off')
        ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='r')

        if predicted_class.lower() in demo.test_generator.filenames[wrong_guess_indices[i]]:
            ax[i // 3, i % 3].set_title("Predicted:{}".format(predicted_class), color='g')
    print('Updated Prediction:')
    display(plt)

    train_button.disabled = False


clear_output()

out = widgets.Output(layout={'border': '1px solid black'})

model_dropdown = widgets.Dropdown(options=['ResNet50', 'MobileNet v2'],
                                  value='ResNet50',
                                  description='Model:')
ngraph_dropdown = widgets.Dropdown(options=ngraph_backends,
                                   value=ngraph_backends[0],
                                   description='nGraph:')
batch_slider = widgets.IntSlider(min=1, max=100, value=batch_size, description='Batch Size:')
epoch_slider = widgets.IntSlider(min=1, max=16, value=epochs, description='Epochs:')

train_button = widgets.Button(description='Train', disabled=False)

training_tab = widgets.VBox(
    children=[model_dropdown, ngraph_dropdown, batch_slider, epoch_slider, train_button])
#fine_tuning_tab = widgets.VBox(children=[
#        batch_slider, epoch_slider])#, fine_tune_slider, fine_tune_button, classify_button,
#        #shuffle_button])
#fine_tuning_tab.disabled = True
tab = widgets.Tab(children=[training_tab])  #, fine_tuning_tab])
tab.set_title(0, 'Training')
#tab.set_title(1, 'Fine Tuning')

progress_bar = widgets.IntProgress()
progress_text = widgets.Label()
epoch_text = widgets.Label()
progress_box = widgets.HBox([progress_bar, progress_text, epoch_text])

gui = widgets.VBox(children=[tab, progress_box])

display(gui)

model_dropdown.observe(on_model_change)
ngraph_dropdown.observe(on_ngraph_change)
train_button.on_click(on_train_clicked)
#self.shuffle_button.on_click(self.init_model)
