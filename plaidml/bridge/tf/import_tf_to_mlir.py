import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.pywrap_mlir import import_graphdef

parser = argparse.ArgumentParser(
    description='Read a TF model and produce serialized MLIR for that model')
parser.add_argument('--model-type',
                    choices=[
                        'keras-resnet',
                        'tfhub-bert',
                        'tfhub-i3d-kin',
                        'tfhub-inception-resnet-v2',
                        'tfhub-retinanet50',
                    ],
                    help='model type and source')
parser.add_argument('--src', help='path to source file(s)')
parser.add_argument('--dst', help='path to write output to')
parser.add_argument('--out-layer-names', help='override the default output layers')
parser.add_argument('--in-layer-names', nargs='*', help='override the default input layers')
parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
parser.add_argument('--optimize-for-inference',
                    action='store_true',
                    help='enable inference optimizations for GraphDef models')
parser.add_argument('--pipeline',
                    help='MLIR pass pipeline',
                    default=','.join([
                        'tf-standard-pipeline',
                        'builtin.func(tf-optimize)',
                        'tf-to-hlo-pipeline',
                        'builtin.func(hlo-legalize-to-linalg)',
                    ]))
args = parser.parse_args()

if args.model_type == 'keras-resnet':
    # Load ResNet50 from Keras
    if args.src:
        print("Warning: --src specified but unused by Keras ResNet50")
    if args.out_layer_names:
        print("Warning: --out-layer-names specified but unused by Keras ResNet50")
    if args.in_layer_names:
        print("Warning: --in-layer-names specified but unused by Keras ResNet50")
    if args.optimize_for_inference:
        print("Warning: --optimize-for-inference specified but unused by Keras ResNet50")
    dst_path = args.dst or "/home/tim/tmp/rn50_tf.mlir"  # TODO: Change the path
    model = tf.keras.applications.resnet50.ResNet50()
    input_shape = [1, 224, 224, 3]  # NHWC for RN
    input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32)]

    @tf.function(input_signature=input_signature)
    def predict(inp):
        return model.predict_step(inp)
elif args.model_type == 'tfhub-bert':
    # For use with the model at https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
    if args.out_layer_names:
        print("Warning: --out-layer-names specified but unused by TFHub BERT")
    if args.in_layer_names:
        print("Warning: --in-layer-names specified but unused by TFHub BERT")
    if args.optimize_for_inference:
        print("Warning: --optimize-for-inference specified but unused by TFHub BERT")
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
elif args.model_type == 'tfhub-i3d-kin':
    # For use with the model at https://tfhub.dev/deepmind/i3d-kinetics-400/1
    if args.out_layer_names:
        print("Warning: --out-layer-names specified but unused by TFHub i3d-kinetics")
    if args.in_layer_names:
        print("Warning: --in-layer-names specified but unused by TFHub i3d-kinetics")
    if args.optimize_for_inference:
        print("Warning: --optimize-for-inference specified but unused by TFHub i3d-kinetics")
    src_dir = args.src or "/home/tim/tmp/tf_hub_models/i3d-kinetics"  # TODO: Change the path
    dst_path = args.dst or "/home/tim/tmp/i3d_tf_todo.mlir"  # TODO: Change the path
    model = tf.saved_model.load(src_dir, tags=[])
    if args.verbose:
        print("Model: ", model)
        print("Signatures?: ", model.signatures)
    # Shape: [batch_size, frame_count, height=224, width=224, 3] for i3d-kin
    input_shape = [1, 16, 224, 224, 3]
    input_signature = [
        tf.TensorSpec(input_shape, tf.float32),
    ]

    @tf.function(input_signature=input_signature)
    def predict(inp):
        return model.signatures['default'](inp)
elif args.model_type == 'tfhub-retinanet50':
    # For use with the model at https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1
    if args.out_layer_names:
        print("Warning: --out-layer-names specified but unused by TFHub retinanet50")
    if args.in_layer_names:
        print("Warning: --in-layer-names specified but unused by TFHub retinanet50")
    if args.optimize_for_inference:
        print("Warning: --optimize-for-inference specified but unused by TFHub retinanet50")
    src_dir = args.src or "/home/tim/tmp/tf_hub_models/retinanet50_1024"  # TODO: Change the path
    dst_path = args.dst or "/home/tim/tmp/retinanet50_tf_todo.mlir"  # TODO: Change the path
    model = tf.saved_model.load(src_dir)
    if args.verbose:
        print("Model: ", model)
        print("Signatures?: ", model.signatures)
    input_shape = [1, 1024, 1024, 3]
    input_signature = [
        tf.TensorSpec(input_shape, tf.uint8),
    ]

    # TODO: Might be multiple outputs? May need to specify?
    @tf.function(input_signature=input_signature)
    def predict(inp):
        return model.signatures['serving_default'](inp)
elif args.model_type == 'tfhub-inception-resnet-v2':
    # For use with the model at https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
    if args.out_layer_names:
        print("Warning: --out-layer-names specified but unused by TFHub inception-resnet-v2")
    if args.in_layer_names:
        print("Warning: --in-layer-names specified but unused by TFHub inception-resnet-v2")
    if args.optimize_for_inference:
        print(
            "Warning: --optimize-for-inference specified but unused by TFHub inception-resnet-v2")
    src_dir = args.src or "/home/tim/tmp/tf_hub_models/inception_resnet_v2"  # TODO: Change the path
    dst_path = args.dst or "/home/tim/tmp/inception_resnet_v2_tf_todo.mlir"  # TODO: Change the path
    model = tf.saved_model.load(src_dir)
    if args.verbose:
        print("Model: ", model)
        print("Signatures?: ", model.signatures)
    input_shape = [1, 1024, 1024, 3]
    input_signature = [
        tf.TensorSpec(input_shape, tf.float32),
    ]

    # TODO: Might be multiple outputs? May need to specify?
    @tf.function(input_signature=input_signature)
    def predict(inp):
        return model.signatures['default'](inp)
else:
    raise ValueError("Invalid --model-type specified")

concrete_fcn = predict.get_concrete_function(*input_signature)
if args.verbose:
    print("Here's the initial concrete function from a saved_model: ", concrete_fcn)  # TODO
concrete_fcn = convert_variables_to_constants_v2(concrete_fcn)  # freeze vars, fixing shapes

with open(dst_path, 'w') as f:
    f.write(tf.mlir.experimental.convert_function(concrete_fcn, pass_pipeline=args.pipeline))

if args.verbose:
    print("Done.")
