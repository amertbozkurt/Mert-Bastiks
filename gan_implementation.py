import tensorflow as tf
import numpy as np


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef


def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])


def style_mod(x, dlatent, **kwargs):
    style = apply_bias(dense(dlatent, fmaps=x.shape[1]*2, gain=1, **kwargs))
    print(style.shape)
    style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
    print(style.shape)
    return x * (style[:,0] + 1) + style[:,1]


def AdaIN(x,y):
    ys , yb = y
    return ys*(x-np.mean(x)*np.ones(x.shape))/np.std(x) + yb


