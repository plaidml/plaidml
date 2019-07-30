class Op(object):

    def __init__(self, params):
        self.params = params
        self.data = None

    def flops(self):
        return None

    def get_dataset(self):
        if self.data is None:
            self.data = getattr(self, 'create_dataset_{}'.format(self.params.backend_name))()
        return self.data

    def build_model(self):
        return getattr(self, 'build_model_{}'.format(self.params.backend_name))()

    def create_dataset_plaid(self):
        raise NotImplementedError

    def build_model_plaid(self):
        raise NotImplementedError

    def get_tc_cache(self):
        raise NotImplementedError

    def create_dataset_tc(self):
        raise NotImplementedError

    def build_model_tc(self):
        raise NotImplementedError

    def create_dataset_tvm(self):
        raise NotImplementedError

    def build_model_tvm(self):
        pass
