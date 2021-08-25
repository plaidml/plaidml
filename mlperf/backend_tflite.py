"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

try:
    # try dedicated tflite package first
    import tflite_runtime
    import tflite_runtime.interpreter as tflite
    _version = tflite_runtime.__version__
    _git_version = tflite_runtime.__git_version__
except:
    # fall back to tflite bundled in tensorflow
    import tensorflow as tf
    from tensorflow.lite.python import interpreter as tflite
    _version = tf.__version__
    _git_version = tf.__git_version__

import backend


class BackendTflite(backend.Backend):

    def __init__(self):
        super(BackendTflite, self).__init__()
        self.sess = None
        self.lock = Lock()

    def version(self):
        return _version + "/" + _git_version

    def name(self):
        return "tflite"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        self.sess = tflite.Interpreter(model_path=model_path)
        self.sess.allocate_tensors()
        # keep input/output name to index mapping
        self.input2index = {i["name"]: i["index"] for i in self.sess.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.sess.get_output_details()}
        # keep input/output names
        self.inputs = list(self.input2index.keys())
        self.outputs = list(self.output2index.keys())
        return self

    def predict(self, feed):
        self.lock.acquire()
        # set inputs
        for k, v in self.input2index.items():
            self.sess.set_tensor(v, feed[k])
        self.sess.invoke()
        # get results
        res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]
        self.lock.release()
        return res
