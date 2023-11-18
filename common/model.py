import tensorflow.math as tf_math

from tensorflow import convert_to_tensor
from tensorflow.keras.losses import Loss
class AngleError(Loss):
    def __init__(self):
        super().__init__(name='angle_error')
    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        return tf_math.reduce_mean(tf_math.abs(tf_math.atan2(tf_math.sin(y_true - y_pred), tf_math.cos(y_true - y_pred))))

from tensorflow import Tensor
def convert_to_real_with_angle(z: Tensor) -> Tensor:
    return tf_math.angle(z)

def add_custom_objects_to_keras():
    from tensorflow.keras.utils import get_custom_objects
    get_custom_objects().update({'angle_error': AngleError, 'convert_to_real_with_angle': convert_to_real_with_angle})

import gc
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback
class FreeMemory(Callback):
    def __init__(self, log_freq=None):
        super().__init__()
        self.log_freq = log_freq
    def on_epoch_end(self, epoch, logs=None):
        if self.log_freq and epoch % self.log_freq == 0:
            print(f'epoch {epoch} ended, info: {logs}')
        clear_session()
        gc.collect()
