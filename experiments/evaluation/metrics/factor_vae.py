# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the disentanglement metric from the FactorVAE paper.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).

******************************************************
RB : This file has been updated to fit into this repository.  
In particular, .gin configurables were mostly removed, with arguments fed explictly via function calls 
Please inform of any licensing issues.
******************************************************

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
    "factor_vae_score",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_factor_vae(ground_truth_data,
		       representation_function,
		       random_state,
		       prune=0.05,
		       batch_size=gin.REQUIRED,
		       num_train=gin.REQUIRED,
		       num_eval=gin.REQUIRED,
		       num_variance_estimate=gin.REQUIRED):
  """Computes the FactorVAE disentanglement metric.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  logging.info("Computing global variances to standardise.")
  #ground_truth_data = ground_truth_data[ground_truth_data.size[0] - ground_truth_data.size[0] % batch_size]
  global_variances = np.squeeze(_compute_variances(ground_truth_data,
					representation_function,
					num_variance_estimate, random_state))
  active_dims = _prune_dims(global_variances, threshold=prune)
  scores_dict = {}

  #if not active_dims.any():
        #active_dims = _prune_dims(global_variances, threshold=0.01)
  #if not active_dims.any():
  #  active_dims = _prune_dims(global_variances, threshold=0.001)
  #if not active_dims.any():
    #active_dims = _prune_dims(global_variances, threshold=0.0)
      
    #scores_dict["train_accuracy"] = 0.
    #scores_dict["eval_accuracy"] = 0.
    #scores_dict["num_active_dims"] = 0
    #return scores_dict

  logging.info("Generating training set.")
  training_votes = _generate_training_batch(
      ground_truth_data, representation_function, batch_size, num_train,
      random_state, global_variances, active_dims)
  classifier = np.argmax(training_votes, axis=0)
  other_index = np.arange(training_votes.shape[1])

  logging.info("Evaluate training set accuracy.")
  train_accuracy = np.sum(
      training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
  logging.info("Training set accuracy: %.2g", train_accuracy)

  logging.info("Generating evaluation set.")
  eval_votes = _generate_training_batch(
      ground_truth_data, representation_function, batch_size, num_eval,
      random_state, global_variances, active_dims)

  logging.info("Evaluate evaluation set accuracy.")
  eval_accuracy = np.sum(
      eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
  logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  scores_dict["num_active_dims"] = len(active_dims)
  return scores_dict


@gin.configurable(
    "prune_dims",
    blacklist=["variances"])
def _prune_dims(variances, threshold=0.):
  """Mask for dimensions collapsed to the prior."""
  print()
  print("Pruning Dims | thresholds ", threshold)
  scale_z = np.sqrt(variances)
  print(scale_z >= threshold)
  print()
  return scale_z >= threshold


def _compute_variances(ground_truth_data, representation_function, batch_size,
		       random_state):
  """Computes the variance for each dimension of the representation.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the variances.
    random_state: Numpy random state used for randomness.

  Returns:
    Vector with the variance of each dimension.
  """
  observations = ground_truth_data.sample_observations(batch_size, random_state)
  representations = representation_function(observations)
  return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(ground_truth_data, representation_function,
			      batch_size, random_state, global_variances,
			      active_dims):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    factor_index: Index of factor coordinate to be used.
    argmin: Index of representation coordinate with the least variance.
  """
  # Select random coordinate to keep fixed.
  factor_index = random_state.randint(ground_truth_data.num_factors)
  # Sample two mini batches of latent variables.
  factors = ground_truth_data.sample_factors(batch_size, random_state)
  # Fix the selected factor across mini-batch.
  factors[:, factor_index] = factors[0, factor_index]
  # Obtain the observations.
  observations = ground_truth_data.sample_observations_from_factors(
      factors, random_state)
  representations = np.squeeze(representation_function(observations))
  local_variances = np.var(representations, axis=0, ddof=1)
  argmin = np.argmin(
      local_variances[active_dims] / global_variances[active_dims]
      )
  return factor_index, argmin


def _generate_training_batch(ground_truth_data, representation_function,
			     batch_size, num_points, random_state,
			     global_variances, active_dims):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
  """
  votes = np.zeros(
      (ground_truth_data.num_factors, global_variances.shape[0]),
      dtype=np.int64)
  for jjj in range(num_points):
    if jjj%1000 == 0:
      print(jjj)
    factor_index, argmin = _generate_training_sample(
	ground_truth_data, representation_function, batch_size, random_state,
	global_variances, active_dims)
    votes[factor_index, argmin] += 1
  return votes
