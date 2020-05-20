import tensorflow as tf
from keras import backend as K
from keras.layers import (
    Layer,
    Activation,
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
from keras.losses import binary_crossentropy, mse
from utils.utils import Resize, BatchNorm


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


LATENT_DIM = 50
INPUT_SHAPE = (240, 320, 3)


def get_encoder(padding="same"):
    inputs = Input(shape=INPUT_SHAPE)

    # block 1
    x = Conv2D(32, kernel_size=(3, 3), padding=padding, strides=2, activation="relu")(
        inputs
    )

    # block 2
    x = Conv2D(64, (3, 3), padding=padding, strides=2, activation="relu")(x)

    # block 3
    x = Conv2D(128, kernel_size=(3, 3), padding=padding, strides=2, activation="relu")(
        x
    )

    # block 4
    x = Conv2D(256, (3, 3), padding=padding, strides=2, activation="relu")(x)

    # block 5
    x = Conv2D(512, kernel_size=(3, 3), padding=padding, strides=2, activation="relu")(
        x
    )

    shape = K.int_shape(x)
    x = Flatten()(x)

    z_mean = Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = Dense(LATENT_DIM, name="z_log_var")(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(LATENT_DIM,), name="z")([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder, shape


def get_decoder(shape, padding="same"):
    latent_inputs = Input(shape=(LATENT_DIM,), name="z_sampling")
    x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # block 5
    x = Conv2DTranspose(
        512, kernel_size=(3, 3), padding=padding, strides=2, activation="relu"
    )(x)

    # block 4
    x = Conv2DTranspose(256, (3, 3), padding=padding, strides=2, activation="relu")(x)

    # block 3
    x = Conv2DTranspose(
        128, kernel_size=(3, 3), padding=padding, strides=2, activation="relu"
    )(x)

    # block 2
    x = Conv2DTranspose(64, (3, 3), padding=padding, strides=2, activation="relu")(x)

    # block 1
    x = Conv2DTranspose(
        32, kernel_size=(3, 3), padding=padding, strides=2, activation="relu"
    )(x)

    outputs = Conv2DTranspose(
        filters=INPUT_SHAPE[-1],
        kernel_size=(3, 3),
        activation="sigmoid",
        padding=padding,
        name="decoder_output",
    )(x)

    outputs = Resize(new_size=INPUT_SHAPE[:2], method="bilinear")(outputs)
    decoder = Model(latent_inputs, outputs, name="decoder")
    return decoder


def get_vae(encoder, decoder):
    inputs = Input(shape=INPUT_SHAPE)
    z_mean, z_log_var, z = encoder(inputs)
    outputs = decoder(z)
    vae = Model(inputs, outputs, name="vae")

    # add metrics to keep track of average z
    for idx in range(z.shape[1]):
        vae.add_metric(z[:, idx], name=f"avg_z_{idx}")

    # get vae loss function
    def vae_loss_fn(y_true, y_pred):
        bce_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        bce_loss *= INPUT_SHAPE[0] * INPUT_SHAPE[1]
        # kl divergence from standard normal
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(bce_loss + kl_loss, axis=-1)

    return vae, vae_loss_fn


def get_classifier(num_classes):
    inputs = Input(shape=(LATENT_DIM,))

    x = Dense(LATENT_DIM)(inputs)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(LATENT_DIM)(x)
    x = BatchNorm()(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)
    outputs = Activation("sigmoid")(x)
    return Model(inputs, outputs, name="classifier")


def vae_network(num_classes, final_activation_fn="sigmoid"):
    inputs = Input(shape=INPUT_SHAPE)
    z = ENCODER(inputs)[2]

    classifier = get_classifier(num_classes)

    outputs = classifier(z)
    return Model(inputs, outputs, name="vae_network")


ENCODER, DECONV_SHAPE = get_encoder()
DECODER = get_decoder(shape=DECONV_SHAPE)
VAE, VAE_LOSS = get_vae(ENCODER, DECODER)


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
