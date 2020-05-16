import tensorflow as tf
from keras import backend as K
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    Reshape,
)
from keras.models import Model, Sequential
import keras


def BatchNorm():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)


class Resize(keras.layers.Layer):
    """ Custom Keras layer that resizes to a new size using interpolation.
    Bypasses the use of Keras Lambda layer
    Args:
      - new_size: tuple, new size to which layer needs to be resized to. Must be (height, width)
      - method: str, method of interpolation to be used. If None, defaults to bilinear.
               Choose amongst 'bilinear', 'nearest', 'lanczos3', 'lanczos5', 'area', 'gaussian', 'mitchellcubic'
    Returns:
      - keras.layers.Layer of size [?, new_size[0], new_size[1], depth]
    """

    def __init__(self, new_size, method="bilinear", **kwargs):
        self.new_size = new_size
        self.method = method
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Resize, self).build(input_shape)

    def call(self, inputs, **kwargs):
        resized_height, resized_width = self.new_size
        return tf.image.resize(
            images=inputs,
            size=[resized_height, resized_width],
            method=self.method,
            # align_corners=True,
        )

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Resize, self).get_config()
        config["new_size"] = self.new_size
        return config


def vanilla_cnn(input_shape, num_classes, final_activation_fn="sigmoid"):

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


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


LATENT_DIM = 20
INPUT_SHAPE = (240, 320, 3)


def get_encoder(input_shape, latent_dim, padding="same"):
    inputs = Input(shape=input_shape)

    # block 1
    x = Conv2D(32, kernel_size=(3, 3), padding=padding, strides=2)(inputs)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 2
    x = Conv2D(64, (3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 3
    x = Conv2D(128, kernel_size=(3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    # block 4
    x = Conv2D(256, (3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    # block 5
    x = Conv2D(512, kernel_size=(3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    shape = K.int_shape(x)
    x = Flatten()(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    print("Encoder model summary")
    encoder.summary()
    return encoder, shape


def get_decoder(latent_dim, shape, padding="same"):
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # block 5
    x = Conv2DTranspose(512, kernel_size=(3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)

    # block 4
    x = Conv2DTranspose(256, (3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)

    # block 3
    x = Conv2DTranspose(128, kernel_size=(3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)

    # block 2
    x = Conv2DTranspose(64, (3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)

    # block 1
    x = Conv2DTranspose(32, kernel_size=(3, 3), padding=padding, strides=2)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)

    outputs = Conv2DTranspose(
        filters=INPUT_SHAPE[-1],
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same",
        name="decoder_output",
    )(x)

    outputs = Resize(new_size=INPUT_SHAPE[:2], method="bilinear")(outputs)

    decoder = Model(latent_inputs, outputs, name="decoder")
    decoder.summary()
    return decoder


def get_vae(encoder, decoder, input_shape):
    inputs = Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(inputs)
    outputs = decoder(z)
    vae = Model(inputs, outputs, name="vae")
    # get vae loss
    bce_loss = K.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    bce_loss *= input_shape[0] * input_shape[1]
    # kl divergence from standard normal
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(bce_loss + kl_loss)
    return vae, vae_loss


ENCODER, DECONV_SHAPE = get_encoder(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)
DECODER = get_decoder(latent_dim=LATENT_DIM, shape=DECONV_SHAPE)
VAE, VAE_LOSS = get_vae(ENCODER, DECODER, input_shape=INPUT_SHAPE)


def vae_network(input_shape, num_classes, final_activation_fn="sigmoid"):
    assert input_shape == INPUT_SHAPE
    inputs = Input(shape=input_shape)
    z = ENCODER(inputs)[2]

    x = Dense(LATENT_DIM)(z)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(LATENT_DIM)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(num_classes)(x)
    outputs = Activation(final_activation_fn)(x)
    return Model(inputs, outputs, name="vae_network")


def landmark_network(input_shape, num_classes, final_activation_fn="sigmoid"):

    input_layer = Input(shape=input_shape)

    # Dense layers
    x = Dense(256)(input_layer)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    # x = Dropout(0.25)(x)

    x = Dense(128)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(num_classes)(x)
    x = BatchNorm()(x)
    x = Activation(final_activation_fn)(x)

    return Model(input_layer, x)
