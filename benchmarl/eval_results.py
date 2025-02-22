#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import collections
import json
from os import walk
from pathlib import Path
from typing import Dict, List, Optional

from marl_eval.plotting_tools.plotting import (
    aggregate_scores,
    performance_profiles,
    plot_single_task,
    probability_of_improvement,
    sample_efficiency_curves,
)
from marl_eval.utils.data_processing_utils import (
    create_matrices_for_rliable,
    data_process_pipeline,
)

from matplotlib import pyplot as plt


def get_raw_dict_from_multirun_folder(multirun_folder: str) -> Dict:
    """Get the ``marl-eval`` input dictionary from the folder of a hydra multirun.

    Examples:
        .. code-block:: python

            from benchmarl.eval_results import get_raw_dict_from_multirun_folder, Plotting
            raw_dict = get_raw_dict_from_multirun_folder(
                multirun_folder="some_prefix/multirun/2023-09-22/17-21-34"
            )
            processed_data = Plotting.process_data(raw_dict)

    Args:
        multirun_folder (str): the absolute path to the multirun folder

    Returns:
        the dict obtained by merging all the json files in the multirun

    """
    return load_and_merge_json_dicts(_get_json_files_from_multirun(multirun_folder))


def _get_json_files_from_multirun(multirun_folder: str) -> List[str]:
    files = []
    for dirpath, _, filenames in walk(multirun_folder):
        for file_name in filenames:
            if file_name.endswith(".json") and "wandb" not in file_name:
                files.append(str(Path(dirpath) / Path(file_name)))
    return files


def load_and_merge_json_dicts(
    json_input_files: List[str], json_output_file: Optional[str] = None
) -> Dict:
    """Loads and merges json dictionaries to form the ``marl-eval`` input dictionary .

    Args:
       json_input_files (list of str): a list containing the absolute paths to the json files
       json_output_file (str, optional): if specified, the merged dictionary will be also written
            to the file in this absolute path

    Returns:
        the dict obtained by merging all the json files

    """

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    dicts = []
    for file in json_input_files:
        if "wandb" not in file and task in file:
            with open(file, "r") as f:
                dicts.append(json.load(f))
    full_dict = {}
    for single_dict in dicts:
        update(full_dict, single_dict)

    if json_output_file is not None:
        with open(json_output_file, "w+") as f:
            json.dump(full_dict, f, indent=4)

    return full_dict


class Plotting:
    """Class containing static utilities for plotting in ``marl-eval``."""

    METRICS_TO_NORMALIZE = ["return"]
    METRIC_TO_PLOT = "return"

    @staticmethod
    def process_data(raw_data: Dict) -> Dict:
        """Call ``data_process_pipeline`` to normalize the chosen metrics and to clean the data

        Args:
            raw_data (dict): the input data

        Returns:
            the processed dict

        """

        return data_process_pipeline(
            raw_data=raw_data, metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE
        )

    @staticmethod
    def create_matrices(processed_data, env_name: str):
        return create_matrices_for_rliable(
            data_dictionary=processed_data,
            environment_name=env_name,
            metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE,
        )

    ############################
    # Environment level plotting
    ############################

    @staticmethod
    def performance_profile_figure(environment_comparison_matrix):
        return performance_profiles(
            environment_comparison_matrix,
            metric_name=Plotting.METRIC_TO_PLOT,
            metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE,
        )

    @staticmethod
    def aggregate_scores(environment_comparison_matrix):
        return aggregate_scores(
            dictionary=environment_comparison_matrix,
            metric_name=Plotting.METRIC_TO_PLOT,
            metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE,
            save_tabular_as_latex=True,
        )

    @staticmethod
    def probability_of_improvement(
        environment_comparison_matrix, algorithms_to_compare: List[List[str]]
    ):
        return probability_of_improvement(
            environment_comparison_matrix,
            metric_name=Plotting.METRIC_TO_PLOT,
            metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE,
            algorithms_to_compare=algorithms_to_compare,
        )

    @staticmethod
    def environemnt_sample_efficiency_curves(sample_effeciency_matrix):
        return sample_efficiency_curves(
            dictionary=sample_effeciency_matrix,
            metric_name=Plotting.METRIC_TO_PLOT,
            metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE,
        )

    ############################
    # Task level plotting
    ############################

    @staticmethod
    def task_sample_efficiency_curves(processed_data, task, env):
        return plot_single_task(
            processed_data=processed_data,
            environment_name=env,
            task_name=task,
            metric_name="return",
            metrics_to_normalize=Plotting.METRICS_TO_NORMALIZE,
        )


if __name__ == "__main__":
    task = "give_way"

    # Define the source and destination paths
    # source_folder = rf"C:\Users\giova\Documents\Research\BenchMARL\multirun\{task}"
    source_folder = rf"C:\Users\giova\Documents\Research\BenchMARL\multirun\17-52-23"

    raw_dict = get_raw_dict_from_multirun_folder(multirun_folder=source_folder)

    processed_data = Plotting.process_data(raw_dict)
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name="vmas")

    Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.aggregate_scores(
        environment_comparison_matrix=environment_comparison_matrix
    )
    Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix
    )

    """if (
        task != "complete"
        and task != "navigation_and_flocking"
        and task != "navigation_and_give_way"
    ):
        Plotting.task_sample_efficiency_curves(
            processed_data=processed_data, env="vmas", task=task
        )"""

    plt.show()
