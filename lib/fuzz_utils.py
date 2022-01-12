
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
"""Utilities for the fuzzer library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random as random
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import lib.dataset as mnist


def basic_mnist_input_corpus(choose_randomly=False, data_dir="/tmp/mnist"):
    """Returns the first image and label from MNIST.

    Args:
      choose_randomly: a boolean indicating whether to choose randomly.
      data_dir: a string giving the location of the original MNIST data.
    Returns:
      A single image and a single label.
    """

    (dataset, _) = tfds.load('mnist',
                             split=['train', 'test'],
                             shuffle_files=True,
                             as_supervised=False,
                             with_info=False)


    examples = dataset.take(10)
    for sample in examples:
        image, label = sample["image"].numpy(), sample["label"].numpy()

    return image, label

def imsave(image, path):
    """Saves an image to a given path.

    This function has the side-effect of writing to disk.

    Args:
        image: The Numpy array representing the image.
        path: A Filepath.
    """
    image = np.squeeze(image)
    with tf.io.gfile.GFile(path, mode="w") as fptr:
        scipy.misc.imsave(fptr, image)


def build_feed_dict(input_tensors, input_batches):
    """Constructs a feed_dict to pass to the run method of TensorFlow Session.

    In the logic we assume all tensors should have the same batch size.
    However, we have to do some crazy stuff to deal with the case when
    some of the tensors have concrete shapes and some don't, especially
    when we're constructing the seed corpus.

    Args:
        input_tensors: The TF tensors into which we will feed the fuzzed inputs.
        input_batches: Numpy arrays that will be fed into the input tensors.

    Returns:
        The feed_dict described above.
    """

    return 0


def get_tensors_from_checkpoint(sess, checkpoint_dir):
    """Loads and returns the fuzzing tensors given a session and a directory.

    It's assumed that the checkpoint directory has checkpoints from a TensorFlow
    model, and moreoever that those checkpoints have 3 collections:
    1. input_tensors: The tensors into which we will feed the fuzzed inputs.
    2. coverage_tensors: The tensors from which we will fetch information needed
      to compute the coverage. The coverage will be used to guide the fuzzing
      process.
    3. metadata_tensors: The tensors from which we will fetch information needed
      to compute the metadata. The metadata can be used for computing the fuzzing
      objective or just to track the progress of fuzzing.

    Args:
      sess: a TensorFlow Session object.
      checkpoint_dir: a directory containing the TensorFlow checkpoints.

    Returns:
        The 3 lists of tensorflow tensors described above.
    """

    
    return 0


def fetch_function(
    model, input_batch
):
    """Fetches from the TensorFlow runtime given inputs.

    Args:
      model: Keras Model object.
      input_batch: numpy arrays we feed to model.

    Returns:
        Coverage list of numpy arrays.
    """
    
    coverage_batch = model.predict_on_batch(input_batch)

    return coverage_batch
