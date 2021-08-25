"""
pytoch native backend 
"""
# pylint: disable=unused-argument,missing-docstring
import torch  # currently supports pytorch1.0
import backend


class BackendPytorchNative(backend.Backend):

    def __init__(self):
        super(BackendPytorchNative, self).__init__()
        self.sess = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.eval()
        # find inputs from the model if not passed in by config
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)
        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)

        # prepare the backend
        self.model = self.model.to(self.device)
        return self

    def predict(self, feed):
        key = [key for key in feed.keys()][0]
        feed[key] = torch.tensor(feed[key]).float().to(self.device)
        with torch.no_grad():
            output = self.model(feed[key])
        return output
