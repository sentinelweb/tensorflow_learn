import base
import tensorflow as tf
import pandas as pd

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
  #
  # Your code here: normalize the inputs.
  #
  processed_features = pd.DataFrame()
  processed_features["latitude"] = base.linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = base.linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = base.linear_scale(examples_dataframe["housing_median_age"])
  processed_features["total_rooms"] = base.linear_scale(examples_dataframe["total_rooms"])
  processed_features["total_bedrooms"] = base.linear_scale(examples_dataframe["total_bedrooms"])
  processed_features["population"] = base.linear_scale(examples_dataframe["population"])
  processed_features["households"] = base.linear_scale(examples_dataframe["households"])
  processed_features["median_income"] = base.linear_scale(examples_dataframe["median_income"])
  processed_features["rooms_per_person"] = base.linear_scale(examples_dataframe["rooms_per_person"])
  return processed_features

normalized_dataframe = normalize_linear_scale(base.preprocess_features(base.california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

_ = base.train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=base.training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=base.validation_targets)

base.multi_page(base.currentdir + "/task1.pdf")