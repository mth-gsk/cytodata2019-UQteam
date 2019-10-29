from keras.applications import mobilenet_v2
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Input, Flatten, Dense

def simple_model(num_classes, input_dim=(None, None), paddingType="valid"):

    model = mobilenet_v2.MobileNetV2(input_shape=[*input_dim, 2], classes=1000)

    model.pop()

    model.add(Dense(num_classes))

    return model