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
"""Fuzz a neural network to find disagreements between normal and quantized."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
from lib import fuzz_utils
from lib.corpus import InputCorpus
from lib.corpus import seed_corpus_from_numpy_arrays
from lib.coverage_functions import raw_logit_coverage_function
from lib.fuzzer import Fuzzer
from lib.mutation_functions import do_basic_mutations
from lib.sample_functions import recent_sample_function
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir",  type=str)
parser.add_argument("--output_path",  type=str)
parser.add_argument("--total_inputs_to_fuzz", type=int, default=100)
parser.add_argument("--mutations_per_corpus_item", type=int, default=100)
parser.add_argument("--perturbation_constraint", type=float)
parser.add_argument("--ann_threshold", type=float, default=1.0)
parser.add_argument("--random_seed_corpus", type=bool, default=False)
FLAGS = parser.parse_args()


def metadata_function(metadata_batches):
    """Gets the metadata."""
    logit_32_batch = metadata_batches[0]
    logit_16_batch = metadata_batches[1]
    metadata_list = []
    for idx in range(logit_16_batch.shape[0]):  
        metadata_list.append((logit_32_batch[idx], logit_16_batch[idx]))
    return metadata_list


def objective_function(corpus_element):
    """Checks if the element is misclassified."""
    logits = corpus_element.coverage
    # prediction_16 = np.argmax(logits_16)
    prediction = np.argmax(logits)
    if prediction == 11:
        print("LOGITS ", logits)
        print("9 Found")
        return True

    # tf.compat.v1.logging.info(
    #     "Objective function satisfied: 32: %s, 16: %s",
    #     prediction_32,
    #     prediction_16,
    # )
    return False


# pylint: disable=too-many-locals
def main():
    """Constructs the fuzzer and fuzzes."""

    # Log more
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    coverage_function = raw_logit_coverage_function
    image, label = fuzz_utils.basic_mnist_input_corpus(
        choose_randomly=FLAGS.random_seed_corpus
    )

    image = np.reshape(image, (28, 28))
    image = (np.expand_dims(image,0))
    print("image shape", image.shape)
    imageio.imwrite("/tmp/input.png", image[0, :, :])
    numpy_arrays = [image]
    print("NUMY ARRAYS ", numpy_arrays[0].shape)

    model = tf.keras.models.load_model(FLAGS.checkpoint_dir)

    size = FLAGS.mutations_per_corpus_item

    def mutation_function(elt):
        """Mutates the element in question."""
        return do_basic_mutations(elt, size, FLAGS.perturbation_constraint)

    seed_corpus = seed_corpus_from_numpy_arrays(
        numpy_arrays, coverage_function, metadata_function, model
    )

    corpus = InputCorpus(
        seed_corpus, recent_sample_function, FLAGS.ann_threshold, "kdtree"
    )

    fuzzer = Fuzzer(
            corpus, 
            model,
            coverage_function,
            metadata_function,
            objective_function,
            mutation_function,
    )

    result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)

    out_image = result.data
    print(out_image.shape)
    imageio.imwrite("/tmp/output.png", out_image)


    return 0


if __name__ == "__main__":
    main()
