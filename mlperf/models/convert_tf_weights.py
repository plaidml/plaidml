import torch

import re
from collections import OrderedDict


def remap_tf_base_names(orig_weights):
    prefix = "backbone."

    # convs
    weights = {k: v for k, v in orig_weights.items() if "FeatureExtractor/MobilenetV1" in k}
    convs = {k: v for k, v in weights.items() if "batchnorm" not in k and "pointwise_" not in k}

    matcher = re.compile("(.*)Conv2d_(\d+)")
    mapping = {}
    for k in convs.keys():
        l = matcher.match(k).group(2)
        name = "pointwise" if "pointwise" in k else "depthwise"
        if l == "0":
            name = "0"
        mapping[k] = "{}{}.{}.weight".format(prefix, l, name)

    # batch norm
    weights = {
        k: v for k, v in orig_weights.items() if "FeatureExtractor/MobilenetV1/MobilenetV1" in k
    }
    weights = {k: v for k, v in weights.items() if "pointwise_" not in k}
    for k in weights.keys():
        l = matcher.match(k).group(2)
        name = "pointwise" if "pointwise" in k else "depthwise"
        op = "scale" if "mul" in k else "bias"
        if l == "0":
            name = "0"
        mapping[k] = "{}{}.{}/BatchNorm.{}".format(prefix, l, name, op)

    return mapping


def remap_tf_extras(orig_weights):
    prefix = "extras."

    weights = {k: v for k, v in orig_weights.items() if "FeatureExtractor/MobilenetV1" in k}
    weights = {k: v for k, v in weights.items() if "pointwise_" in k}

    matcher = re.compile("(.*)Conv2d_(\d+)_(\d)x(\d)")
    mapping = {}
    for k in weights.keys():
        m = matcher.match(k)
        l = int(m.group(2)) - 2
        ks = int(m.group(3))
        if ks == 1:
            pos = 0
        else:
            pos = 2
        wtype = "weight" if "weight" in k else "bias"
        mapping[k] = "{}{}.{}.{}".format(prefix, l, pos, wtype)

    return mapping


def remap_tf_predictors(orig_weights):
    mapping = {}

    # regression
    weights = {k: v for k, v in orig_weights.items() if "BoxPredictor" in k}
    weights = {k: v for k, v in weights.items() if "BoxEncodingPredictor" in k}

    matcher = re.compile("BoxPredictor_(\d+)")
    for k in weights.keys():
        pos = matcher.match(k).group(1)
        wtype = "weight" if "weights" in k else "bias"
        mapping[k] = "predictors.{}.regression.{}".format(pos, wtype)

    # classification
    weights = {k: v for k, v in orig_weights.items() if "BoxPredictor" in k}
    weights = {k: v for k, v in weights.items() if "ClassPredictor" in k}

    for k in weights.keys():
        pos = matcher.match(k).group(1)
        wtype = "weight" if "weights" in k else "bias"
        mapping[k] = "predictors.{}.classification.{}".format(pos, wtype)

    return mapping


def remap_tf_names(weights):
    layers_base = remap_tf_base_names(weights)
    layers_extra = remap_tf_extras(weights)
    layers_predictors = remap_tf_predictors(weights)

    layers = {}
    layers.update(layers_base)
    layers.update(layers_extra)
    layers.update(layers_predictors)

    return layers


def get_state_dict(weights):
    layers = remap_tf_names(weights)
    state_dict = OrderedDict()

    for orig, new in layers.items():
        weight = weights[orig]
        weight = torch.as_tensor(weight, dtype=torch.float32)
        if weight.dim() == 4:
            p = (2, 3, 0, 1)
            if "pointwise" in orig or "backbone.0." in new or "BoxPredictor" in orig:
                p = (3, 2, 0, 1)
            weight = weight.permute(*p).contiguous()
        state_dict[new] = weight
    return state_dict


def read_tf_weights(frozen_model):
    import tensorflow as tf
    from tensorflow.python.framework import tensor_util
    weights = {}
    with tf.Session() as sess:
        with tf.gfile.GFile(frozen_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
        for n in graph_def.node:
            if n.op == 'Const':
                weights[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)
    return weights
