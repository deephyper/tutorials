import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from dh_project.mnist.load_data import load_data_train_valid

HERE = os.path.dirname(os.path.abspath(__file__))


def run(config: dict) -> float:
    verbose = 1

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: write logs on worker 0.
    verbose = verbose if hvd.rank() == 0 else 0

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    (train_X, train_y), (valid_X, valid_y) = load_data_train_valid()

    def to_dataset(X, y):
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(X[..., tf.newaxis], tf.float32),
                tf.cast(y, tf.int64),
            )
        )
        return dataset

    train_dataset = to_dataset(train_X, train_y).repeat().shuffle(10000).batch(128)
    valid_dataset = to_dataset(valid_X, valid_y).batch(128).repeat()
    valid_steps_per_epoch = len(valid_X) // 128

    mnist_model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
            tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = config["lr"] * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True
    )

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    mnist_model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=scaled_lr, warmup_epochs=3, verbose=verbose
        ),
    ]

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    history = mnist_model.fit(
        train_dataset,
        steps_per_epoch=500 // hvd.size(),
        callbacks=callbacks,
        epochs=3,
        validation_data=valid_dataset,
        validation_steps=valid_steps_per_epoch,
        verbose=verbose,
    )

    return history.history["val_accuracy"][-1]