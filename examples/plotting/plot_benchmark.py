#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
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
    tasks2 = [VmasTask.NAVIGATION.get_from_yaml()]
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        QmixConfig.get_from_yaml(),
    ]
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks2,
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
    folder_results = "../../multirun/complete"
    tasks_names = [
        # "flocking",
        # "give_way",
        "navigation",
    ]
    algo_names = ["iql", "causaliql", "qmix", "causalqmix", "vdn", "causalvdn"]

    experiment_json_files = []

    # Iterate through the directory structure
    for root, dirs, files in os.walk(folder_results):
        for file in files:
            # Check if the file is a JSON file
            if file.endswith(".json"):
                # Check if the file name contains any combination of tasks_names and algo_names
                if any(task in file for task in tasks_names) and any(
                    algo in file for algo in algo_names
                ):
                    # If both conditions are met, add the file to the list with full path
                    experiment_json_files.append(os.path.join(root, file))

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
    """Plotting.task_sample_efficiency_curves(
        processed_data=processed_data, env="vmas", task=tasks_names
    )"""

    try:
        Plotting.probability_of_improvement(
            environment_comparison_matrix,
            algorithms_to_compare=[algo_names],
        )
    except Exception as e:
        pass
    plt.show()
