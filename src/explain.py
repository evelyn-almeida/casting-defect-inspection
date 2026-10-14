import numpy as np


def grad_cam(model, images, layer="last_conv"):
    """Grad-CAM heatmaps for the 'defective' score, resized to the input size."""
    import tensorflow as tf
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer).output, model.output])
    x = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        conv, pred = grad_model(x, training=False)
        score = pred[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1, keepdims=True))
    cam = tf.image.resize(cam, images.shape[1:3]).numpy()[..., 0]
    return cam / (cam.max(axis=(1, 2), keepdims=True) + 1e-9)
