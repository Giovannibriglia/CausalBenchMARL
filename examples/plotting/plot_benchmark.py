#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import json
import os
from pathlib import Path
from typing import List

from benchmarl.eval_results import load_and_merge_json_dicts, Plotting

from matplotlib import pyplot as plt


def run_benchmark() -> List[str]:
    from benchmarl.algorithms import MappoConfig, QmixConfig
    from benchmarl.benchmark import Benchmark
    from benchmarl.environments import VmasTask
    from benchmarl.experiment import ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    # Configure experiment
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    experiment_config.loggers = []
    experiment_config.max_n_iters = 100

    # Configure benchmark
    tasks = [VmasTask.NAVIGATION.get_from_yaml()]
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        QmixConfig.get_from_yaml(),
    ]
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds={0, 1},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )

    # For each experiment, run it and get its output file name
    experiments = benchmark.get_experiments()
    experiments_json_files = []
    for experiment in experiments:
        exp_json_file = str(
            Path(experiment.folder_name) / Path(experiment.name + ".json")
        )
        experiments_json_files.append(exp_json_file)
        experiment.run()
    return experiments_json_files


if __name__ == "__main__":
    experiment_json_files = []
    folder_path = "../running/"
    key1 = "causaliql_give_way_causalmlp"
    key2 = "iql_give_way_mlp"

    # Walk through the folder and subfolders
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if (file_name.endswith(".json") and key1 in file_name) or (
                file_name.endswith(".json") and key2 in file_name
            ):
                # Get the full file path
                full_path = os.path.join(root, file_name)
                experiment_json_files.append(full_path)

    # Uncomment this to rerun the benchmark that generates the files
    # experiment_json_files = run_benchmark()

    raw_dict = load_and_merge_json_dicts(experiment_json_files)

    processed_data = Plotting.process_data(raw_dict)
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name="vmas")

    # Plotting
    Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.aggregate_scores(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix
    )
    Plotting.task_sample_efficiency_curves(
        processed_data=processed_data, env="vmas", task="give_way"
    )
    try:
        Plotting.probability_of_improvement(
            environment_comparison_matrix,
            algorithms_to_compare=[["causaliql", "iql"]],
        )
    except:
        pass
    plt.show()
