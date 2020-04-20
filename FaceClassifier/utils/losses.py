import numpy as np
import tensorflow as tf
import keras.backend as K

def tversky_loss(y_true, y_pred, beta=0.5):
  numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
  return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  epsilon = K.epsilon()
  y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
  loss = -alpha * K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred)
  return K.mean(loss, axis=-1)

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  epsilon = K.epsilon()
  # clip to prevent NaN's and Inf's
  pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
  pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
  return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1),axis=-1) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0),axis=-1)

