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
                        'mlperf-bert',
                        'mlperf-bert-experimental',
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

if args.model_type == 'keras-resnet' or args.model_type == 'tfhub-bert':
    from_saved_model = True
else:
    from_saved_model = False

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
    # Load from file for BERT
    if args.out_layer_names:
        print("Warning: --out-layer-names specified but unused by TFHub BERT")
    if args.in_layer_names:
        print("Warning: --in-layer-names specified but unused by TFHub BERT")
    if args.optimize_for_inference:
        print("Warning: --optimize-for-inference specified but unused by TFHub BERT")
    src_dir = args.src or "/home/tim/tmp/tf_hub_models/bert/resaved"  # TODO: Change the path
    dst_path = args.dst or "/home/tim/tmp/bert_tf_todo.mlir"  # TODO: Change the path
    model = tf.saved_model.load(src_dir)
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
elif args.model_type == 'mlperf-bert-experimental':
    # Load BERT trying to use import_graphdef
    # Experimental paths for loading from *.pb
    src_dir = args.src or "/home/tim/mlcommons/inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pb"  # TODO: Change the path
    dst_path = args.dst or "/home/tim/tmp/bert_tf_todo.mlir"  # TODO: Change the path
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(src_dir, 'rb') as f:
        graph_def.ParseFromString(f.read())
    inputs = args.in_layer_names or [
        'input_ids:0',
        'input_mask:0',
        'segment_ids:0',
    ]
    outputs = args.out_layer_names or ['input_ids:0']
    if args.verbose:
        print("Layer names:")
        print([n.name for n in graph_def.node])
    # with open(dst_path, 'w') as f:
    #     # graph_def = tf.compat.v1.import_graph_def(graph_def, name="")
    #     f.write(tf.mlir.experimental.convert_graph_def(graph_def, pass_pipeline=args.pipeline))
    # if args.verbose:
    #     print("Done with convert_graph_def")
    #     raise RuntimeError("Forced Abort")

    mlir_tf = import_graphdef(
        graph_def,
        args.pipeline,
        False,
        input_names=[],  #[item.split(':')[0] for item in inputs],
        input_data_types=[],  #["DT_INT", "DT_INT", "DT_INT"],
        input_data_shapes=[],  #["1,256", "1,256", "1,256"],
        output_names=["logits:0"])  #[item.split(':')[0] for item in outputs])
    with open(dst_path, 'w') as f:
        f.write(mlir_tf)
    if args.verbose:
        print("Done with experimental section...")
elif args.model_type == 'mlperf-bert':
    # Experimental paths for loading from *.pb
    src_dir = args.src or "/home/tim/mlcommons/inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pb"  # TODO: Change the path
    dst_path = args.dst or "/home/tim/tmp/bert_tf_todo.mlir"  # TODO: Change the path
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(src_dir, 'rb') as f:
        graph_def.ParseFromString(f.read())
    if args.verbose:
        print("Layer names:")
        print([n.name for n in graph_def.node])
    inputs = args.in_layer_names or [
        'input_ids:0',
        'input_mask:0',
        'segment_ids:0',
    ]
    outputs = args.out_layer_names or ['input_ids:0']
    if args.optimize_for_inference:
        optimized_graph_def = optimize_for_inference(graph_def,
                                                     [item.split(':')[0] for item in inputs],
                                                     [item.split(':')[0] for item in outputs],
                                                     tf.dtypes.int32.as_datatype_enum, False)
        graph_def = optimized_graph_def

    def wrap_frozen_graph(graph_def, inputs, outputs):

        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

    concrete_fcn = wrap_frozen_graph(graph_def, inputs, outputs)
    if args.verbose:
        print("Here's the concrete function (in theory): ", concrete_fcn)  # TODO
else:
    raise ValueError("Invalid --model-type specified")

if from_saved_model:
    concrete_fcn = predict.get_concrete_function(*input_signature)
    if args.verbose:
        print("Here's the initial concrete function from a saved_model: ", concrete_fcn)  # TODO
    concrete_fcn = convert_variables_to_constants_v2(concrete_fcn)  # freeze vars, fixing shapes
else:
    pass
    # concrete_fcn = wrap_frozen_graph()

with open(dst_path, 'w') as f:
    f.write(tf.mlir.experimental.convert_function(concrete_fcn, pass_pipeline=args.pipeline))

if args.verbose:
    print("Done.")
