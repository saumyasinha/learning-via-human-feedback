from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.models import Model, Sequential

def BatchNorm():
  return BatchNormalization(momentum=0.95, epsilon=1e-5)


def build_model(input_shape, num_classes, final_activation_fn='softmax'):

  input_layer = Input(shape=input_shape)
  
  # block 1
  x = Conv2D(16, (3,3), padding='same', use_bias=False)(input_layer)
  x = Activation('relu')(x)
  x = BatchNorm()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

  # block 2
  x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
  x = Activation('relu')(x)
  x = BatchNorm()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

  # block 3
  x = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
  x = Activation('relu')(x)
  x = BatchNorm()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x)

  # block 4
  x = Conv2D(128, (3,3), padding='same', use_bias=False)(x)
  x = Activation('relu')(x)
  x = BatchNorm()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.4)(x)


  x = Flatten()(x)

  # Dense layers
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = BatchNorm()(x)
  x = Dropout(0.4)(x)

  x = Dense(num_classes)(x)
  x = Activation(final_activation_fn)(x)

  return Model(input_layer,x)