"""
null backend
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import time

import backend


class BackendNull(backend.Backend):

    def __init__(self):
        super(BackendNull, self).__init__()

    def version(self):
        return "-"

    def name(self):
        return "null"

    def image_format(self):
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        self.outputs = ["output"]
        self.inputs = ["input"]
        return self

    def predict(self, feed):
        # yield to give the thread that feeds our queue a chance to run
        time.sleep(0)
        # return something fake
        return [[0]]
