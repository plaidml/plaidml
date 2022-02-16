"""
abstract backend class
"""

# pylint: disable=unused-argument,missing-docstring


class Backend():

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def version(self):
        raise NotImplementedError("Backend:version")

    def name(self):
        raise NotImplementedError("Backend:name")

    def load(self, model_path, inputs=None, outputs=None):
        raise NotImplementedError("Backend:load")

    def predict(self, feed):
        raise NotImplementedError("Backend:predict")
