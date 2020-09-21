# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
import numpy as np
import tensorflow as tf

import i3d

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': '../kinetics-i3d/data/v_CricketShot_g04_c01_rgb.npy',
    'flow': '../kinetics-i3d/data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': '../kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '../kinetics-i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '../kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '../kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '../kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = '../kinetics-i3d/data/label_map.txt'
_LABEL_MAP_PATH_600 = '../kinetics-i3d/data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(tf.float32,
                                   shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(NUM_CLASSES,
                                         spatial_squeeze=True,
                                         final_endpoint='Logits')
            rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(
                        ':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(tf.float32,
                                    shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(NUM_CLASSES,
                                          spatial_squeeze=True,
                                          final_endpoint='Logits')
            flow_logits, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')
            rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
            tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            feed_dict[rgb_input] = rgb_sample

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')
            flow_sample = np.load(_SAMPLE_PATHS['flow'])
            tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
            feed_dict[flow_input] = flow_sample

        out_logits, out_predictions = sess.run([model_logits, model_predictions],
                                               feed_dict=feed_dict)

        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]

        print('Norm of logits: %f' % np.linalg.norm(out_logits))
        print('\nTop classes and probabilities')
        for index in sorted_indices[:20]:
            print(out_predictions[index], out_logits[index], kinetics_classes[index])


if __name__ == '__main__':
    tf.app.run(main)
