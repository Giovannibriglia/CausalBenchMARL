#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import CausalIqlConfig
from benchmarl.environments.vmas.common import MaskVmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.causal_mlp import CausalMlpConfig


if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = MaskVmasTask.GIVE_WAY.get_from_yaml()
    task_name = task.name
    # Loads from "benchmarl/conf/algorithm/mappo.yaml"0.
    algorithm_config = CausalIqlConfig.get_from_yaml()

    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    CausalMlpConfig.task = task
    model_config = CausalMlpConfig.get_from_yaml()
    model_config.task_name = task_name
    critic_model_config = MlpConfig.get_from_yaml()

    seed_input = int(input("Set seed: "))

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed_input,
        config=experiment_config,
    )
    experiment.run()
