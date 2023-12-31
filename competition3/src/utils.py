import tensorflow as tf


def check_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
