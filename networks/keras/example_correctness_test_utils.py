import keras.callbacks


# Saves a single set of loss info for correctness testing
class TrainingHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.loss = None
        self.acc = None
        self.val_loss = None
        self.val_acc = None

    def on_epoch_end(self, epoch, logs={}):
        self.loss = logs.get('loss')
        self.acc = logs.get('acc')
        self.val_loss = logs.get('val_loss')
        self.val_acc = logs.get('val_acc')


# Saves each epoch's loss info for correctness testing
class TrainingHistoryList(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.loss = list()
        self.acc = list()
        self.val_loss = list()
        self.val_acc = list()

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


# Runs compile and execution stopwatches at appropriate times in training
class StopwatchManager(keras.callbacks.Callback):

    def __init__(self, execution_stop_watch, compile_stop_watch):
        self.csw = compile_stop_watch
        self.esw = execution_stop_watch

    def on_train_begin(self, logs={}):
        self.csw.start()

    def on_batch_end(self, batch, logs={}):
        if self.csw.is_running():
            self.csw.stop()
            self.esw.start()

    def on_train_end(self, logs={}):
        self.esw.stop()
