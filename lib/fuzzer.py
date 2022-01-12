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
"""Defines the actual Fuzzer object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.corpus import CorpusElement
import tensorflow as tf


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(
        self,
        corpus,
        model,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
    ):
        """Init the class.

    Args:
      corpus: An InputCorpus object.
      coverage_function: a function that does CorpusElement -> Coverage.
      metadata_function: a function that does CorpusElement -> Metadata.
      objective_function: a function that checks if a CorpusElement satisifies
        the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
      mutation_function: a function that does CorpusElement -> Metadata.
      fetch_function: grabs numpy arrays from the TF runtime using the relevant
        tensors.
    Returns:
      Initialized object.
    """
        self.corpus = corpus
        self.model = model
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function

    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""

        for iteration in range(iterations):
            if iteration % 100 == 0:
                tf.compat.v1.logging.info("fuzzing iteration: %s", iteration)
            parent = self.corpus.sample_input()

            # Get a mutated batch for each input tensor
            mutated_data_batches = self.mutation_function(parent)

            # Grab the coverage for mutated batch from the TF runtime.
            coverage_batches = self.model.predict(mutated_data_batches)
            # print("COVERAGE BATHES SHAPE", coverage_batches.shape)

            # # Get the coverage - one from each batch element
            # mutated_coverage_list = self.coverage_function(coverage_batches)
            # print("MUTATED COV LIST SHAPE ", len(mutated_coverage_list))

            # Check for new coverage and create new corpus elements if necessary.
            # pylint: disable=consider-using-enumerate
            for idx in range(coverage_batches.shape[0]):
                new_element = CorpusElement(
                    mutated_data_batches[idx],
                    #[batch[idx] for batch in mutated_data_batches],
                    coverage_batches[idx],
                    parent,
                )
                if self.objective_function(new_element):
                    return new_element
                self.corpus.maybe_add_to_corpus(new_element)

        return None
