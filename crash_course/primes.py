# Testing docker install
import tensorflow as tf
with tf.Graph().as_default():
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
    ones = tf.ones([6], dtype=tf.int32)
    just_beyond_primes = tf.add(primes, ones)
    with tf.Session() as sess:
        print just_beyond_primes.eval()