import json
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch

LABEL_reward_action_values = "reward_action_values"


class CausalActionsFilter:
    def __init__(self, ci_online: bool, task: str, **kwargs):
        self.ci_online = ci_online
        self.task_name = task.lower()

        self.last_obs_continuous = None
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.path_best = self.get_original_script_path()

        causal_table = pd.read_pickle(f"{self.path_best}/causal_table.pkl")
        with open(f"{self.path_best}/best_others.json", "r") as file:
            info = json.load(file)

        self.indexes_to_discr = [
            str(s).replace("agent_0_obs_", "")
            for s in info["discrete_intervals"]
            if "reward" not in s and "value" not in s and "kind" not in s
        ]
        self.n_groups = (
            None if info["grouped_features"][0] == 0 else info["grouped_features"][0]
        )
        self.indexes_to_group = (
            None
            if self.n_groups is None
            else [
                int(str(s).replace("agent_0_obs_", ""))
                for s in info["grouped_features"][1]
            ]
        )
        self.n_actions = None
        self._define_action_mask_inputs(causal_table)

    def get_original_script_path(self):
        try:
            # If Hydra is used, get the original working directory
            original_wd = hydra.utils.get_original_cwd()

            # Construct the path to causality_best/task_name
            causality_best_path = (
                Path(original_wd)
                / "benchmarl"
                / "models"
                / "causality_best"
                / self.task_name
            )

        except ValueError:
            # If Hydra is not used, fallback to the current working directory
            original_path = os.getcwd()
            causality_best_path = str(original_path).replace(
                "examples\\running",
                f"benchmarl\\models\\causality_best\\{self.task_name}",
            )

        return causality_best_path

    def _define_action_mask_inputs(self, causal_table: pd.DataFrame):
        def actions_mask_filter(
            reward_action_values, possible_rewards, possible_actions
        ):
            old_min, old_max = min(possible_rewards), max(possible_rewards)
            averaged_mean_dict = {float(action): 0.0 for action in possible_actions}

            for reward_value, action_probs in reward_action_values.items():
                rescaled_value = (float(reward_value) - old_min) / (old_max - old_min)
                for action, prob in action_probs.items():
                    averaged_mean_dict[float(action)] += prob * rescaled_value

            num_entries = len(reward_action_values)
            averaged_mean_dict = {
                action: value / num_entries
                for action, value in averaged_mean_dict.items()
            }

            values = list(averaged_mean_dict.values())
            percentile_25 = np.percentile(values, 25)
            actions_mask = torch.tensor(
                [0 if value <= percentile_25 else 1 for value in values],
                device=self.device,
            )

            if actions_mask.sum() == 0:
                actions_mask = torch.tensor(
                    [0 if value <= 0 else 1 for value in values], device=self.device
                )

            return actions_mask

        def process_rav(values):
            possible_rewards = [float(key) for key in values[0].keys()]
            possible_actions = [
                float(key) for key in values[0][str(possible_rewards[0])].keys()
            ]
            self.n_actions = len(possible_actions)
            return [
                actions_mask_filter(val, possible_rewards, possible_actions)
                for val in values
            ]

        def df_to_tensors(df: pd.DataFrame):
            return {
                str(key).replace("agent_0_obs_", ""): (
                    torch.tensor(
                        columns_values.values, dtype=torch.float32, device=self.device
                    )
                    if key != LABEL_reward_action_values
                    else process_rav(columns_values.values)
                )
                for key, columns_values in df.items()
            }

        self.dict_causal_table_tensors = df_to_tensors(causal_table)
        self.action_masks_from_causality = self.dict_causal_table_tensors[
            LABEL_reward_action_values
        ]

        self.indexes_obs_in_causal_table = [
            causal_table.columns.get_loc(s)
            for s in causal_table.columns
            if s != LABEL_reward_action_values
        ]
        self.values_obs_in_causal_table = torch.stack(
            [
                value
                for key, value in self.dict_causal_table_tensors.items()
                if key != LABEL_reward_action_values
            ]
        )
        self.values_obs_in_causal_table_expanded = (
            self.values_obs_in_causal_table.unsqueeze(0)
        )

        ok_indexes_obs = []
        if any("agent_0_obs" in col for col in causal_table.columns):
            ok_indexes_obs.extend(
                [
                    int(str(s).replace("agent_0_obs_", ""))
                    for s in causal_table.columns
                    if "agent_0_obs" in s
                ]
            )

        if any("kind" in col for col in causal_table.columns) or any(
            "value" in col for col in causal_table.columns
        ):
            start_group = len(self.indexes_to_discr)
            if self.n_groups is not None:
                for n in range(self.n_groups):
                    index = start_group + n * 2
                    ok_indexes_obs.extend([index, index + 1])

        self.ok_indexes_obs = torch.tensor(ok_indexes_obs, device=self.device)

    def get_action_mask(self, multiple_observation: torch.Tensor):
        # Validate input in a single step
        if not isinstance(multiple_observation, torch.Tensor):
            raise ValueError("multiple_observation must be a tensor")

        # Batch process delta observation for continuous values
        def calculate_delta_obs_continuous(
            current_obs: torch.Tensor, last_obs: torch.Tensor
        ):
            # return torch.sub(current_obs, last_obs, out=current_obs)
            assert current_obs.shape == last_obs.shape, print(
                current_obs.shape, last_obs.shape
            )
            return current_obs - last_obs

        def group_obs(obs):
            mask_ok = torch.ones(obs.shape[1], dtype=bool, device=self.device)

            # Now apply the mask correctly
            mask_ok[self.indexes_to_group] = False

            values_ok = obs[:, mask_ok]
            values_to_group = obs[:, ~mask_ok]

            # Batch process top-k to reduce redundant computation
            top_values, top_indices = torch.topk(values_to_group, self.n_groups, dim=1)
            index_value_pairs = torch.stack((top_indices.float(), top_values), dim=-1)

            return torch.cat(
                (values_ok.flatten(1), index_value_pairs.flatten(1)), dim=1
            )

        def discretize_obs(obs):
            # Select only relevant parts of the observation to avoid broadcasting unnecessary data
            obs_values = obs[:, self.ok_indexes_obs].unsqueeze(2)

            # We only expand once for the necessary batch size
            batch_size = obs_values.size(0)

            # Expand values_obs_in_causal_table_expanded to match the second dimension of closest_indices
            values_obs_in_causal_table_expanded = (
                self.values_obs_in_causal_table_expanded.expand(
                    batch_size, obs_values.size(1), -1
                )
            )

            # Optimize the differences calculation using in-place operations where possible
            differences = torch.abs_(obs_values - values_obs_in_causal_table_expanded)

            # Perform argmin operation more efficiently, keeping operations on the same device
            closest_indices = torch.argmin(differences, dim=2)

            # Use gather with values_obs_in_causal_table_expanded and closest_indices
            discretized_values = values_obs_in_causal_table_expanded.gather(
                2, closest_indices.unsqueeze(2)
            ).squeeze(2)

            # Create a tensor with the same shape as the original `obs`
            discretized = torch.zeros_like(obs)

            # Insert the discretized values back into the original observation shape
            discretized[:, self.ok_indexes_obs] = discretized_values

            return discretized

        def compute_action_mask(obs):
            batch_size = obs.shape[0]  # Get the batch size

            # Initialize a tensor for the action masks, defaulting to all ones
            action_masks = torch.ones(batch_size, self.n_actions, device=self.device)

            # Stack all values in the causal table into a tensor
            values_obs_stack = torch.stack(
                [
                    self.values_obs_in_causal_table[j][: obs.shape[1]]
                    for j in range(len(self.indexes_obs_in_causal_table))
                ],
                dim=0,
            )  # Shape: (num_entries_in_causal_table, obs_dim)

            # Compare all observations in the batch with the values_obs_stack in a single operation
            obs_expanded = obs.unsqueeze(0).expand(
                values_obs_stack.shape[0], batch_size, obs.shape[1]
            )  # Shape: (num_entries_in_causal_table, batch_size, obs_dim)
            comparison = (values_obs_stack.unsqueeze(1) == obs_expanded).all(
                dim=-1
            )  # Shape: (num_entries_in_causal_table, batch_size)

            # Find the first valid index for each observation in the batch
            valid_indices = comparison.float().argmax(dim=0)  # Shape: (batch_size)

            # Create a mask for the batch, marking where valid indices are found
            valid_mask = comparison.sum(dim=0) > 0  # Shape: (batch_size)

            # Apply the action masks from the valid indices for those that matched
            if valid_mask.any():  # Ensure there are valid indices to avoid errors
                selected_action_masks = self.action_masks_from_causality[
                    valid_indices[valid_mask]
                ].to(self.device)
                action_masks[valid_mask] = selected_action_masks

            return action_masks

        def process_obs(obs):
            # print(obs.shape)
            delta_obs_cont = calculate_delta_obs_continuous(
                obs, self.last_obs_continuous
            )
            # print(delta_obs_cont.shape)
            grouped_obs = group_obs(delta_obs_cont) if self.n_groups else delta_obs_cont
            # print(grouped_obs.shape)
            return compute_action_mask(discretize_obs(grouped_obs))

        # Flatten observations for batch processing
        num_envs, num_agents, _ = multiple_observation.shape
        multiple_observation_flatten = multiple_observation.view(
            -1, multiple_observation.size(-1)
        )

        # If last continuous observations are available, process the new observations
        if (
            self.last_obs_continuous is not None
            and multiple_observation_flatten.shape == self.last_obs_continuous.shape
        ):
            # Batch processing for multiple observations
            action_masks = process_obs(multiple_observation_flatten)
        else:
            action_masks = torch.ones(
                (num_envs * num_agents, self.n_actions), device=self.device
            )
        # print("Action mask shape: ", action_masks.shape)
        # Update last observations
        self.last_obs_continuous = multiple_observation_flatten
        return action_masks.view(num_envs, num_agents, -1).bool()

        """num_envs, num_agents, _ = multiple_observation.shape
        return torch.ones(
            (num_envs, num_agents, 9), device=self.device, dtype=torch.bool
        )"""


if __name__ == "__main__":
    online_ci = True
    causal_action_filter = CausalActionsFilter(online_ci, "navigation")
