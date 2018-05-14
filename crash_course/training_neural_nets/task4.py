# https://colab.research.google.com/notebooks/mlcc/improving_neural_net_performance.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=improvingneuralnets-colab&hl=en#scrollTo=1hwaFCE71OPZ
# https://developers.google.com/machine-learning/crash-course/training-neural-networks/programming-exercise
import base
import math
import tensorflow as tf
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)


def normalize_lat_long(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized."""
  processed_features = pd.DataFrame()

  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])

  return processed_features

normalized_dataframe = normalize_lat_long(base.preprocess_features(base.california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

# tried a few variants of this and this worked quite well
# Final RMSE (on training data):   64.49
# Final RMSE (on validation data): 69.21
_, adam_training_losses, adam_validation_losses = base.train_nn_regression_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.005),#0.009
    steps=10000,
    batch_size=500,
    # hidden_units=[100, 100, 20],
    hidden_units=[1000, 100, 50, 50, 50],
    training_examples=normalized_training_examples,
    training_targets=base.training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=base.validation_targets)