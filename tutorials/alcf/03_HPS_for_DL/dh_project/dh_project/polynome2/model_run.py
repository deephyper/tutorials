import numpy as np
import tensorflow as tf
from dh_project.polynome2.load_data import load_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


def r2(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
    output_scores = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.reduce_mean(output_scores)
    return r2


HISTORY = None


def run(point):
    global HISTORY
    (x_train, y_train), (x_valid, y_valid) = load_data()

    if point["activation"] == "identity":
        point["activation"] = None

    model = Sequential()
    model.add(
        Dense(
            point["units"],
            activation=point["activation"],
            input_shape=tuple(np.shape(x_train)[1:]),
        )
    )
    model.add(Dense(1))

    model.summary()

    model.compile(loss="mse", optimizer=RMSprop(lr=point["lr"]), metrics=[r2])

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=1000,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_r2", mode="max", verbose=1, patience=10)],
        validation_data=(x_valid, y_valid),
    )

    HISTORY = history.history

    return history.history["val_r2"][-1]
