import keras


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        object.__init__(self)
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.dic = {}

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.dic = {}

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))

    def on_train_end(self, logs=None):

        self.dic['losses'] = self.losses
        self.dic['val_losses'] = self.val_losses
        self.dic['accuracy'] = self.accuracy
        self.dic['val_accuracy'] = self.val_accuracy
