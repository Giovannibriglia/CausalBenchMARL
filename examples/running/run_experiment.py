#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import  IqlConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.GIVE_WAY.get_from_yaml()

    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    algorithm_config = IqlConfig.get_from_yaml()

    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    seed_input = int(input('Set seed: '))

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed_input,
        config=experiment_config,
    )
    experiment.run()
