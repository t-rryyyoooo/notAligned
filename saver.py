import tensorflow as tf

class LatestWeightSaver(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename_ = filename

    def on_epoch_end(self, epoch, logs):
        self.model.save_weights(self.filename_)


class PeriodicWeightSaver(tf.keras.callbacks.Callback):
    def __init__(self, logdir, interval):
        self.logdir_ = logdir
        self.interval_ = interval

    def on_epoch_end(self, epoch, logs):
        if epoch % self.interval_ == 0:
            filename = self.logdir_ + "/weights_e{:02d}.hdf5".format(epoch)
            self.model.save_weights(filename)
