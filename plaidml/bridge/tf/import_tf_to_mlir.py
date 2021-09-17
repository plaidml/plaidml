import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.pywrap_mlir import import_graphdef


class ModelMetadata(object):
    '''Handles metadata for non-Keras SavedModels

    Includes information about save format, inputs, and default file location.'''

    def __init__(self):
        self._metadata = {}

    def add_model_type(self,
                       name,
                       input_shape,
                       input_dtype,
                       sig_name,
                       default_src=None,
                       default_dst=None,
                       tags=None):
        self._metadata[name] = {
            'default_src': default_src,
            'default_dst': default_dst,
            'input_shape': input_shape,
            'input_dtype': input_dtype,
            'sig_name': sig_name,
            'tags': tags,
        }

    def import_concrete_function(self, model_type, src=None, verbose=False):
        src_dir = src or self._metadata[model_type]['default_src']
        input_signature = [
            tf.TensorSpec(
                self._metadata[model_type]['input_shape'],
                self._metadata[model_type]['input_dtype'],
            )
        ]
        # TODO: I think TF needs me to hold on to this TensorSpec, so I'm sticking it on this object. Not great architecture, rethink later.
        self._input_sig = input_signature
        model = tf.saved_model.load(src_dir, tags=self._metadata[model_type]['tags'])
        if verbose:
            print("Model: ", model)
            print("Signatures: ", model.signatures)

        def run(inp):
            return model.signatures[self._metadata[model_type]['sig_name']](inp)

        # TODO: I think TF needs me to hold on to this function, so I'm sticking it on this object. Not great architecture, revisit later
        self._func = tf.function(run, input_signature=input_signature)
        return self._func.get_concrete_function(*input_signature)

    def default_dst_path(self, model_type):
        return self._metadata[model_type]['default_dst']


def populate_metadata(model_meta):
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/deepmind/i3d-kinetics-400/1
        'tfhub-i3d-kin',
        [1, 16, 224, 224, 3],  # Input Shape -- [batch_size, frame_count, height=224, width=224, 3]
        tf.float32,  # Input DType
        'default',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/i3d-kinetics",
        default_dst="/home/tim/tmp/i3d_tf_todo.mlir",
        tags=[],
    )
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1
        'tfhub-retinanet50',
        [1, 1024, 1024, 3],  # Input Shape
        tf.uint8,  # Input DType
        'serving_default',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/retinanet50_1024",
        default_dst="/home/tim/tmp/retinanet50_tf_todo.mlir",
    )
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
        'tfhub-inception-resnet-v2',
        [1, 1024, 1024, 3],  # Input Shape
        tf.float32,  # Input DType
        'default',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/inception_resnet_v2",
        default_dst="/home/tim/tmp/inception_resnet_v2_tf_todo.mlir",
    )
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/silero/silero-stt/en/1
        'tfhub-silero-stt',
        [1024],  # Input Shape -- this is just a guess at a reasonable size
        tf.float32,  # Input DType
        'serving_default',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/silero_stt",
        default_dst="/home/tim/tmp/silero_stt_tf_todo.mlir",
    )
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/deepmind/mmv/s3d/1
        'tfhub-s3d-video',
        [1, 32, 200, 200, 3],  # Input Shape -- Batch x T x H x W x 3
        tf.float32,  # Input DType
        'video',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/s3d",
        default_dst="/home/tim/tmp/s3d_video_tf_todo.mlir",
        tags=[],
    )
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/deepmind/mmv/s3d/1
        'tfhub-s3d-audio',
        [1, 153600],  # Input Shape -- Batch x T
        tf.float32,  # Input DType
        'audio',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/s3d",
        default_dst="/home/tim/tmp/s3d_audio_tf_todo.mlir",
        tags=[],
    )
    model_meta.add_model_type(
        # For use with the model at https://tfhub.dev/tensorflow/efficientnet/b7/classification/1
        'tfhub-efficientnet-b7',
        [1, 600, 600, 3],  # Input Shape
        tf.float32,  # Input DType
        'serving_default',  # Signature name
        default_src="/home/tim/tmp/tf_hub_models/efficientnet-b7",
        default_dst="/home/tim/tmp/efficientnet_b7_tf_todo.mlir",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read a TF model and produce serialized MLIR for that model')
    parser.add_argument('--model-type',
                        choices=[
                            'keras-resnet',
                            'tfhub-bert',
                            'tfhub-i3d-kin',
                            'tfhub-inception-resnet-v2',
                            'tfhub-retinanet50',
                            'tfhub-silero-stt',
                            'tfhub-s3d-video',
                            'tfhub-s3d-audio',
                            'tfhub-efficientnet-b7',
                        ],
                        help='model type and source')
    parser.add_argument('--src', help='path to source file(s)')
    parser.add_argument('--dst', help='path to write output to')
    parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
    parser.add_argument('--pipeline',
                        help='MLIR pass pipeline',
                        default=','.join([
                            'tf-standard-pipeline',
                            'builtin.func(tf-optimize)',
                            'tf-to-hlo-pipeline',
                            'builtin.func(hlo-legalize-to-linalg)',
                        ]))
    args = parser.parse_args()

    model_meta = ModelMetadata()
    populate_metadata(model_meta)

    if args.model_type == 'keras-resnet':
        # Load ResNet50 from Keras
        if args.src:
            print("Warning: --src specified but unused by Keras ResNet50")
        dst_path = args.dst or "/home/tim/tmp/rn50_tf.mlir"  # TODO: Change the path
        model = tf.keras.applications.resnet50.ResNet50()
        input_shape = [1, 224, 224, 3]  # NHWC for RN
        input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32)]

        @tf.function(input_signature=input_signature)
        def predict(inp):
            return model.predict_step(inp)

        concrete_fcn = predict.get_concrete_function(*input_signature)
    elif args.model_type == 'tfhub-bert':
        # For use with the model at https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
        src_dir = args.src or "/home/tim/tmp/tf_hub_models/bert/resaved"  # TODO: Change the path
        dst_path = args.dst or "/home/tim/tmp/bert_tf_todo.mlir"  # TODO: Change the path
        model = tf.saved_model.load(src_dir)
        if args.verbose:
            print("Model: ", model)
        input_shape = [1, 256]  # [batch_size, seq_length] for BERT
        input_signature = [
            tf.TensorSpec(input_shape, tf.int32),
            tf.TensorSpec(input_shape, tf.int32),
            tf.TensorSpec(input_shape, tf.int32),
        ]

        @tf.function(input_signature=input_signature)
        def predict(type_ids, word_ids, mask):
            return model.__call__(
                {
                    'input_type_ids': type_ids,
                    'input_word_ids': word_ids,
                    'input_mask': mask
                },
                True,  # Or False?
                None
            )['default']  # Use 'sequence_output' to get what seems to be a training version (e.g. has dropout)

        concrete_fcn = predict.get_concrete_function(*input_signature)
    else:
        # Default non-Keras single-input case
        dst_path = args.dst or model_meta.default_dst_path(args.model_type)
        concrete_fcn = model_meta.import_concrete_function(args.model_type,
                                                           src=args.src,
                                                           verbose=args.verbose)

    if args.verbose:
        print("Here's the initial concrete function from a saved_model: ", concrete_fcn)  # TODO
    concrete_fcn = convert_variables_to_constants_v2(concrete_fcn)  # freeze vars, fixing shapes

    with open(dst_path, 'w') as f:
        f.write(tf.mlir.experimental.convert_function(concrete_fcn, pass_pipeline=args.pipeline))

    if args.verbose:
        print("Done.")
