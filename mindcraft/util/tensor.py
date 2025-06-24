import numpy as np
from numpy import squeeze, eye, arange, asarray, argmax


def to_one_hot(a, num_classes, reshape=True):
    shape = a.shape
    y = squeeze(eye(num_classes)[a.reshape(-1)])
    if reshape:
        return y
    return y.reshape((*shape, num_classes))


def to_categorical(a, classes=None, reshape=True):
    a = np.asarray(a)
    if classes is None:
        classes = arange(a.shape[-1])
    elif isinstance(classes, int):
        classes = arange(classes)
    classes = np.asarray(classes)
    num_classes = len(classes)
    out = asarray([classes[c] for c in argmax(a[..., :num_classes], axis=-1)], dtype=classes.dtype)
    if reshape:
        return out
    return out.reshape(a.shape[:-1])
