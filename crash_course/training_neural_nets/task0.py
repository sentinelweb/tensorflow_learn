import base
import tensorflow as tf
import pandas as pd

_ = base.train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=base.training_examples,
    training_targets=base.training_targets,
    validation_examples=base.validation_examples,
    validation_targets=base.validation_targets)

base.multi_page(base.currentdir + "/task0.pdf")