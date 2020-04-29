from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.models import Model, Sequential


def BatchNorm():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)


def custom_model(input_shape, num_classes, final_activation_fn="sigmoid"):

    input_layer = Input(shape=input_shape)

    # block 1
    x = Conv2D(32, kernel_size=(3, 3), padding="valid", use_bias=True)(input_layer)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 2
    # x = Conv2D(32, kernel_size=(3, 3), padding="valid", use_bias=False)(x)
    # x = BatchNorm()(x)
    # x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 3
    x = Conv2D(64, (3, 3), padding="valid", use_bias=False)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # x = Dropout(0.2)(x)

    # block 4
    x = Conv2D(128, kernel_size=(3, 3), padding="valid", use_bias=False)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    # block 5
    x = Conv2D(256, (3, 3), padding="valid", use_bias=False)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    # block 6
    x = Conv2D(512, kernel_size=(3, 3), padding="valid", use_bias=False)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # # block 7
    # x = Conv2D(1024, (3, 3), padding="same", use_bias=False)(x)
    # x = BatchNorm()(x)
    # x = Activation("relu")(x)
    # x = Dropout(0.4)(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)

    # Dense layers
    x = Dense(512)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(512)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(num_classes)(x)
    x = Activation(final_activation_fn)(x)

    return Model(input_layer, x)


def VGG16_model(input_shape, num_classes, final_activation_fn="sigmoid"):

    input_layer = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3),
                activation='relu',
                padding='same')(x)
    x = Conv2D(512, (3, 3),
                activation='relu',
                padding='same')(x)
    x = Conv2D(512, (3, 3),
                activation='relu',
                padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Dense layers
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(num_classes, activation=final_activation_fn)(x)

    model = Model(input_layer, x)
    return model
