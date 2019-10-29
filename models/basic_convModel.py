from functools import wraps, reduce

from keras import backend as K
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Input, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {"kernel_regularizer": l2(5e-4)}
    darknet_conv_kwargs["padding"] = (
        "valid" if kwargs.get("strides") == (2, 2) else "same"
    )
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)



def Conv_DW(filters, *args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {"use_bias": False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(
            filters=filters // 4, kernel_size=(3, 3), *args, **no_bias_kwargs
        ),
        DarknetConv2D(filters=filters, kernel_size=(1, 1), *args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
    )


def simple_model(num_classes, input_dim=(None, None)):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    if len(input_dim) == 2:
        input_dim = (*input_dim, 2)
    inputs = Input(input_dim)
    x1 = compose(
        Conv_DW(14),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv_DW(16),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv_DW(20),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv_DW(32),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv_DW(48),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv_DW(64),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        Conv_DW(128),
        Flatten()
    )(inputs)

    y1 = compose(
        Dense(128),
        Dense(64),
        Dense(num_classes)
    )(x1)

    return Model(inputs, y1)
