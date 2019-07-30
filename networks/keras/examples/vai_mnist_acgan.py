# -*- coding: utf-8 -*-
"""
This code is derived from Keras examples/mnist_acgan.py.

Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
"""
from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

import argparse
import numpy as np
import os
from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar

np.random.seed(1337)

num_classes = 10
batch_size = 32

# Warning: The stop_watch timings are pretty incoherent and in particular don't
# really correspond to compile & execution


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()

    cnn.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
    cnn.add(Reshape((3, 3, 384)))

    # upsample to (7, 7, ...)
    cnn.add(
        Conv2DTranspose(192,
                        5,
                        strides=1,
                        padding='valid',
                        activation='relu',
                        kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    # upsample to (14, 14, ...)
    cnn.add(
        Conv2DTranspose(96,
                        5,
                        strides=2,
                        padding='same',
                        activation='relu',
                        kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    # upsample to (28, 28, ...)
    cnn.add(
        Conv2DTranspose(1,
                        5,
                        strides=2,
                        padding='same',
                        activation='tanh',
                        kernel_initializer='glorot_normal'))

    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(num_classes, latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2, input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])


def build_combined(latent_size, generator, discriminator):
    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    fake, aux = discriminator(fake)
    return Model([latent, image_class], [fake, aux])


def _preprocess_mnist_input(train_input_truncation, test_input_truncation):
    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[:train_input_truncation]
    y_train = y_train[:train_input_truncation]
    x_test = x_test[:test_input_truncation]
    y_test = y_test[:test_input_truncation]

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test


def _visualize_output(
        filename,
        generator,
        latent_size,
        x_train=None,
        y_train=None,
        real_image_offset=0,
):
    if x_train is None and y_train is not None:
        raise ValueError('If y_train is provided to _visualize_output, x_train must be too')
    if y_train is None and x_train is not None:
        raise ValueError('If x_train is provided to _visualize_output, y_train must be too')
    num_rows = 40
    noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)), (num_classes, 1))

    sampled_labels = np.array([[i] * num_rows for i in range(num_classes)]).reshape(-1, 1)

    # get a batch to display
    generated_images = generator.predict([noise, sampled_labels], verbose=0)

    if x_train is not None:
        # prepare real images sorted by class label
        real_labels = y_train[real_image_offset * num_rows * num_classes:(real_image_offset + 1) *
                              num_rows * num_classes]
        indices = np.argsort(real_labels, axis=0)
        real_images = x_train[real_image_offset * num_rows * num_classes:(real_image_offset + 1) *
                              num_rows * num_classes][indices]

        # display generated images, white separator, real images
        img = np.concatenate(
            (generated_images, np.repeat(np.ones_like(x_train[:1]), num_rows,
                                         axis=0), real_images))

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28) for r in np.split(img, 2 * num_classes + 1)],
                              axis=-1) * 127.5 + 127.5).astype(np.uint8)
    else:
        # arrange into a grid
        img = (np.concatenate([r.reshape(-1, 28) for r in np.split(generated_images, num_classes)],
                              axis=-1) * 127.5 + 127.5).astype(np.uint8)

    Image.fromarray(img).save(filename)


def _format_array_comparison(correct, max_rel_err, max_abs_err, fail_ratio, name=None):
    passed = 'passed' if correct else 'FAILED'
    if name is None:
        name = ''
    else:
        name = ' ' + name
    format = 'Test{} {} with max rel. error {:.5}, max abs. error {:.5}, and fail rate {:%}'
    return format.format(name, passed, max_rel_err, max_abs_err, fail_ratio)


def _compare_arrays(x, y, rtol=1e-04, atol=1e-05, verbose=False):
    # copied from analysis.py
    correct = np.allclose(x, y, rtol=rtol, atol=atol)
    if verbose:
        # This duplicates allclose calculation for more detailed report
        relative_error = ((rtol * np.absolute(x - y)) / (atol + rtol * np.absolute(y)))
        max_rel_error = np.amax(relative_error)
        max_abs_error = np.amax(np.absolute(x - y))
        correct_entries = 0
        incorrect_entries = 0
        for x in np.nditer(relative_error):
            if x > rtol:
                incorrect_entries += 1
            else:
                correct_entries += 1
        try:
            fail_ratio = incorrect_entries / float(correct_entries + incorrect_entries)
        except ZeroDivisionError:
            fail_ratio = float('NaN')
        print(_format_array_comparison(correct, max_rel_error, max_abs_error, fail_ratio))
    return correct, max_rel_error, max_abs_error, fail_ratio


def test_models(
        generator,
        discriminator,
        combined,
        x_test,
        y_test,
        latent_size,
        save_file=None,
        compare_file=None,
        reference_weights_gen=None,
        reference_weights_disc=None,  # Destructive
):
    # reference_weights_gen: test the discriminator against a generator with weights from the passed file
    # reference_weights_disc: test the generator against a discriminator with weights from the passed file
    # Warning: using reference_weights_disc rewrites the weights of discriminator (& combined)

    if compare_file is not None or save_file is not None:
        # If working with a golden file, must use standard seed
        np.random.seed(47)

    num_test = x_test.shape[0]
    # generate a new batch of noise
    noise = np.random.uniform(-1, 1, (num_test, latent_size))

    # sample some labels from p_c and generate images from them
    sampled_labels = np.random.randint(0, num_classes, num_test)
    stop_watch.start()
    generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)
    stop_watch.stop()
    x = np.concatenate((x_test, generated_images))

    # Compare against or make a golden output file
    if compare_file is not None or save_file is not None:
        stop_watch.start()
        discriminator_predictions = discriminator.predict(x, verbose=False)
        stop_watch.stop()
        if compare_file is not None:
            with open(compare_file, 'r') as f:
                golden = np.load(f)
                comparisons = [(generated_images, golden['imgs']),
                               (discriminator_predictions[0], golden['is_synth']),
                               (discriminator_predictions[1], golden['cat'])]
                for curr, gold in comparisons:
                    print("Inference tests v. golden:")
                    result = _compare_arrays(curr, gold, verbose=True)
                    if not result[0]:
                        raise RuntimeError("Failed inference correctness: {}".format(
                            _format_array_comparison(*result)))
        if save_file is not None:
            with open(save_file, 'w') as f:
                np.savez(f,
                         imgs=generated_images,
                         is_synth=discriminator_predictions[0],
                         cat=discriminator_predictions[1])

    # If testing discriminator against a reference generator, recreate x from that generator
    if reference_weights_gen is not None:
        ref_gen = build_generator(latent_size)
        ref_gen.load_weights(reference_weights_gen)
        stop_watch.start()
        generated_images = ref_gen.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)
        stop_watch.stop()
        x = np.concatenate((x_test, generated_images))
    y = np.array([1] * num_test + [0] * num_test)
    aux_y = np.concatenate((y_test, sampled_labels), axis=0)

    # see if the discriminator can figure itself out...
    stop_watch.start()
    discriminator_test_loss = discriminator.evaluate(x, [y, aux_y], verbose=False)
    stop_watch.stop()

    # make new noise
    noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
    sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

    trick = np.ones(2 * num_test)

    if reference_weights_disc is not None:
        discriminator.load_weights(reference_weights_disc)

    stop_watch.start()
    generator_test_loss = combined.evaluate([noise, sampled_labels.reshape((-1, 1))],
                                            [trick, sampled_labels],
                                            verbose=False)
    stop_watch.stop()
    return generator_test_loss, discriminator_test_loss


def print_test_report(generator_loss, discriminator_loss, name, metrics_names):
    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *metrics_names))
    print('-' * 65)

    gen_name = 'generator ({})'.format(name)
    disc_name = 'discriminator ({})'.format(name)
    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
    print(ROW_FMT.format(gen_name, *generator_loss))
    print(ROW_FMT.format(disc_name, *discriminator_loss))


def train(
        generator,
        discriminator,
        combined,
        latent_size,
        epochs,
        x_train,
        y_train,
        x_test,
        y_test,
        verbose=True,
        save_weights=True,
        visualize=True,
):
    num_train, num_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {} of {}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
        progress_bar = Progbar(target=num_batches)

        # we don't want the discriminator to also maximize the classification
        # accuracy of the auxiliary classifier on generated images, so we
        # don't train discriminator to produce class labels for generated
        # images (see https://openreview.net/forum?id=rJXTf9Bxg).
        # To preserve sum of sample weights for the auxiliary classifier,
        # we assign sample weight of 2 to the real images.
        disc_sample_weight = [
            np.ones(2 * batch_size),
            np.concatenate((np.ones(batch_size) * 2, np.zeros(batch_size)))
        ]

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, num_classes, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            stop_watch.start()
            generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))],
                                                 verbose=0)
            stop_watch.stop()

            x = np.concatenate((image_batch, generated_images))

            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            stop_watch.start()
            epoch_disc_loss.append(
                discriminator.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight))
            stop_watch.stop()

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size) * soft_one

            stop_watch.start()
            epoch_gen_loss.append(
                combined.train_on_batch([noise, sampled_labels.reshape((-1, 1))],
                                        [trick, sampled_labels]))
            stop_watch.stop()

        progress_bar.update(index + 1)

        # Save & report on training loss
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        if verbose:
            print('  Training loss for epoch {}:'.format(epoch))
            print_test_report(generator_train_loss, discriminator_train_loss, 'train',
                              discriminator.metrics_names)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        # Evaluate the testing loss
        generator_test_loss, discriminator_test_loss = test_models(generator, discriminator,
                                                                   combined, x_test, y_test,
                                                                   latent_size)
        if verbose:
            print('  Testing loss for epoch {}:'.format(epoch))
            print_test_report(generator_test_loss, discriminator_test_loss, 'test',
                              discriminator.metrics_names)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        # save weights every epoch
        if save_weights:
            generator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
            discriminator.save_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(epoch),
                                       True)

        # generate some digits to display
        if visualize:
            _visualize_output('plot_epoch_{0:03d}_generated.png'.format(epoch), generator,
                              latent_size, x_train, y_train, epoch - 1)
    return train_history, test_history


def main(
        epochs=2,
        latent_size=32,
        adam_lr=0.0002,
        adam_beta_1=0.5,
        train_input_truncation=12000,
        test_input_truncation=1000,
        test_training=True,
        test_inference=True,
        verbose=True,
        save_weights=False,
        visualize=False,
        gen_weights_file=None,
        disc_weights_file=None,
        save_golden_file=None,
        load_golden_file=None,
):
    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                          loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)

    # we only want to be able to train generation for the combined model
    # This raises a warning, but it does so in the Keras examples version too
    discriminator.trainable = False

    combined = build_combined(latent_size, generator, discriminator)
    print('Combined model:')
    combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                     loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    combined.summary()

    # Load training & testing data from MNIST
    x_train, y_train, x_test, y_test = _preprocess_mnist_input(train_input_truncation,
                                                               test_input_truncation)

    if test_training:
        # First test against a threshold
        train_history, test_history = train(generator, discriminator, combined, latent_size,
                                            epochs, x_train, y_train, x_test, y_test, verbose,
                                            save_weights, visualize)
        passing = True
        total_loss_thresh = 1.2
        gen_loss_thresh = 1.0
        aux_loss_thresh = 0.5
        for type in ['generator', 'discriminator']:
            if test_history[type][-1][0] > total_loss_thresh:
                passing = False
            if test_history[type][-1][1] > gen_loss_thresh:
                passing = False
            if test_history[type][-1][2] > aux_loss_thresh:
                passing = False
        if not passing:
            print('Requirements: ')
            print_test_report([total_loss_thresh, gen_loss_thresh, aux_loss_thresh],
                              [total_loss_thresh, gen_loss_thresh, aux_loss_thresh], 'req',
                              discriminator.metrics_names)
            print('Actual: ')
            print_test_report(test_history['generator'][-1], test_history['discriminator'][-1],
                              'final', discriminator.metrics_names)
            raise RuntimeError('MNIST AC-GAN failed to train to sufficiently low loss')

        if gen_weights_file is not None and disc_weights_file is not None:
            # Also test against saved models
            gen_loss, disc_loss = test_models(
                generator,
                discriminator,
                combined,
                x_test,
                y_test,
                latent_size,
                reference_weights_gen=gen_weights_file,
                reference_weights_disc=disc_weights_file,
            )
            gen_loss_threshold = 0.5  # Strong generator fools weak discriminator
            disc_loss_threshold = 1.2  # Discriminator will be weak against other generators; ensure not too weak
            if verbose:
                print("  Loss vs. reference networks:")
                print_test_report(gen_loss, disc_loss, 'v. ref', discriminator.metrics_names)
            if gen_loss[0] > gen_loss_threshold or disc_loss[0] > disc_loss_threshold:
                raise RuntimeError(
                    'MNIST AC-GAN failed to sufficiently outperform reference model')
    else:
        train_history = defaultdict(list)
        test_history = defaultdict(list)

    if test_inference:
        if gen_weights_file is None or disc_weights_file is None:
            raise ValueError('Must have weights files to test inference')
        # Test the network using saved weights
        generator.load_weights(gen_weights_file)
        discriminator.load_weights(disc_weights_file)

        generator_test_loss, discriminator_test_loss = test_models(generator,
                                                                   discriminator,
                                                                   combined,
                                                                   x_test,
                                                                   y_test,
                                                                   latent_size,
                                                                   save_file=save_golden_file,
                                                                   compare_file=load_golden_file)

        # Print report on the saved weights results
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        if verbose:
            print('  Loss for model loaded from saved weights:')
            print_test_report(generator_test_loss, discriminator_test_loss, 'saved',
                              discriminator.metrics_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', default=os.path.join('..', 'data', 'acgan_infer_golden.npz'))
    parser.add_argument('--make-golden', default=None)
    parser.add_argument('--gen-weights', default=os.path.join('..', 'data', 'gen_weights.hdf5'))
    parser.add_argument('--disc-weights', default=os.path.join('..', 'data', 'disc_weights.hdf5'))
    parser.add_argument('--input-size-train', type=int, default=30000)
    parser.add_argument('--input-size-test', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=3)
    args, remain = parser.parse_known_args()
    print(args, remain)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    gen_weights_file = args.gen_weights
    disc_weights_file = args.disc_weights
    golden = args.golden
    if not os.path.exists(gen_weights_file):
        gen_weights_file = os.path.normpath(os.path.join(this_dir, gen_weights_file))
    if not os.path.exists(disc_weights_file):
        disc_weights_file = os.path.normpath(os.path.join(this_dir, disc_weights_file))
    if golden is not None and not os.path.exists(golden):
        golden = os.path.normpath(os.path.join(this_dir, golden))
    main(epochs=args.epochs,
         train_input_truncation=args.input_size_train,
         test_input_truncation=args.input_size_test,
         gen_weights_file=gen_weights_file,
         disc_weights_file=disc_weights_file,
         save_golden_file=args.make_golden,
         load_golden_file=golden)
