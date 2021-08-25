"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt

import backend


class BackendOnnxruntime(backend.Backend):

    def __init__(self):
        super(BackendOnnxruntime, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def image_format(self):
        """image_format. For onnx it is always NCHW."""
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        opt = rt.SessionOptions()
        # enable level 3 optimizations
        # FIXME: enable below once onnxruntime 0.5 is released
        # opt.set_graph_optimization_level(3)
        self.sess = rt.InferenceSession(model_path, opt)
        # get input and output names
        if not inputs:
            self.inputs = [meta.name for meta in self.sess.get_inputs()]
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = [meta.name for meta in self.sess.get_outputs()]
        else:
            self.outputs = outputs
        return self

    def predict(self, feed):
        """Run the prediction."""
        return self.sess.run(self.outputs, feed)
