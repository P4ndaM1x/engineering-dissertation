def prepare_data():
    import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    import numpy as np
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
        (x_train, y_train), (x_test, y_test) = prepare_data()
        
        import tensorflow as tf
        import gc
        class FreeMemory(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                tf.keras.backend.clear_session()
                gc.collect()

        raw_model = tf.keras.models.load_model('/host/code/raw_model.keras')
        fit_history = raw_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1, callbacks=[FreeMemory()])
        raw_model.save('/host/code/trained_model.keras')
        
        import pandas as pd
        df = pd.DataFrame(fit_history.history)
        df['epoch'] = fit_history.epoch
        df.to_pickle('/host/code/fit_history.pkl')