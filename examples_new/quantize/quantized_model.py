# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a model and its quantized counterpart."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import lib.dataset as mnist
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir",  type=str, default="/tmp/nanfuzzer")
parser.add_argument("--data_dir", type=str, default="/tmp/mnist")
parser.add_argument("--training_steps", type=int, default=35000)

FLAGS = parser.parse_args()

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def main():
    """Train a model and a sort-of-quantized version."""
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    checkpoint_dir = os.path.dirname(FLAGS.checkpoint_dir)
    model.fit(x_train, y_train, epochs=FLAGS.training_steps)
    model.evaluate(x_test,  y_test, verbose=2)

    model.save(FLAGS.checkpoint_dir)

if __name__ == "__main__":
    main()
