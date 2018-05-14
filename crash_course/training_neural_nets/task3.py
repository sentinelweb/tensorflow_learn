
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

def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))

def normalize_task3_attempted(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
  processed_features = pd.DataFrame()
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
  processed_features["total_rooms"] = log_normalize(examples_dataframe["total_rooms"])
  processed_features["total_bedrooms"] = log_normalize(examples_dataframe["total_bedrooms"])
  processed_features["population"] = clip(log_normalize(examples_dataframe["population"]),4,9)
  processed_features["households"] = log_normalize(examples_dataframe["households"])
  processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
  processed_features["rooms_per_person"] = clip(log_normalize(examples_dataframe["rooms_per_person"]), .2, 1.8)
  return processed_features

def normalize_solution(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized."""
  processed_features = pd.DataFrame()

  processed_features["households"] = log_normalize(examples_dataframe["households"])
  processed_features["median_income"] = log_normalize(examples_dataframe["median_income"])
  processed_features["total_bedrooms"] = log_normalize(examples_dataframe["total_bedrooms"])
  
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])

  processed_features["population"] = linear_scale(clip(examples_dataframe["population"], 0, 5000))
  processed_features["rooms_per_person"] = linear_scale(clip(examples_dataframe["rooms_per_person"], 0, 5))
  processed_features["total_rooms"] = linear_scale(clip(examples_dataframe["total_rooms"], 0, 10000))

  return processed_features

def normalize_solution_clipping(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized."""
  processed_features = pd.DataFrame()

  processed_features["households"] = clip(log_normalize(examples_dataframe["households"]),2,9)
  processed_features["median_income"] = log_normalize(examples_dataframe["median_income"])
  processed_features["total_bedrooms"] = clip(log_normalize(examples_dataframe["total_bedrooms"]),2,9)
  
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])

  processed_features["population"] = linear_scale(clip(examples_dataframe["population"], 0, 5000))
  processed_features["rooms_per_person"] = linear_scale(clip(examples_dataframe["rooms_per_person"], 0, 5))
  processed_features["total_rooms"] = linear_scale(clip(examples_dataframe["total_rooms"], 0, 10000))

  return processed_features

# run the test (normalize_task3_attempted was worse)
for i in range(0,3):
  print "iteration: %s" % i
  plt.figure()
  if i is 0:
    title="Solution - attempted"
    normalized_dataframe = normalize_task3_attempted(base.preprocess_features(base.california_housing_dataframe))
  elif i is 1:
    title="Solution - answer"
    normalized_dataframe = normalize_solution(base.preprocess_features(base.california_housing_dataframe))
  elif i is 2:
    title="Solution - answer more clipping"
    normalized_dataframe = normalize_solution_clipping(base.preprocess_features(base.california_housing_dataframe))
  normalized_training_examples = normalized_dataframe.head(12000)
  normalized_validation_examples = normalized_dataframe.tail(5000)
  plt.title(title)

  # _, adam_training_losses, adam_validation_losses = base.train_nn_regression_model(
  #     my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
  #     steps=500,
  #     batch_size=100,
  #     hidden_units=[10, 10],
  #     training_examples=normalized_training_examples,
  #     training_targets=base.training_targets,
  #     validation_examples=normalized_validation_examples,
  #     validation_targets=base.validation_targets)

  _ = normalized_training_examples.hist(bins=20, figsize=(16, 12), xlabelsize=10)
  
  

base.multi_page(base.currentdir + "/task3.pdf")