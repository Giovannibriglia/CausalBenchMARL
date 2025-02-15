import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Tuple
from stable_baselines3 import PPO, SAC

from vmas import make_env, Wrapper


# Helper function to flatten Tuple space manually
def manual_flatten_space(space):
    if isinstance(space, Tuple):
        # Calculate the total size of the flattened space for all the subspaces in the tuple
        low = np.concatenate([s.low.flatten() for s in space.spaces], axis=0)
        high = np.concatenate([s.high.flatten() for s in space.spaces], axis=0)
        return Box(low=low, high=high, dtype=np.float32)
    else:
        return space


# Helper function to manually flatten observations
def manual_flatten(obs, space):
    if isinstance(space, Tuple):
        # Flatten each observation from each subspace and concatenate
        flattened_obs = np.concatenate([o.flatten() for o in obs], axis=-1)
        return flattened_obs
    else:
        return obs.flatten()


# Helper function to manually unflatten actions
def manual_unflatten(action, space):
    if isinstance(space, Tuple):
        # Unflatten the action according to the original Tuple structure
        sizes = [np.prod(s.shape) for s in space.spaces]
        split_actions = np.split(action, np.cumsum(sizes)[:-1])
        unflattened_action = [
            split.reshape(s.shape) for split, s in zip(split_actions, space.spaces)
        ]
        return tuple(unflattened_action)
    else:
        return action.reshape(space.shape)


# Observation Wrapper
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Manually flatten the observation space if it's a tuple
        self.observation_space = manual_flatten_space(self.env.observation_space)

    def observation(self, obs):
        # Manually flatten the observation
        return manual_flatten(obs, self.env.observation_space)


# Action Wrapper
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Manually flatten the action space if it's a tuple
        self.action_space = manual_flatten_space(self.env.action_space)

    def action(self, action):
        # Manually unflatten the action before passing it to the environment
        return manual_unflatten(action, self.env.action_space)

    def reverse_action(self, action):
        # Manually flatten the action when returning from environment
        return manual_flatten(action, self.env.action_space)


# Reward Wrapper to handle aggregation of multi-agent rewards
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, reward_aggregation="sum"):
        super().__init__(env)
        self.reward_aggregation = reward_aggregation

    def reward(self, rewards):
        # Aggregate rewards
        if isinstance(rewards, list) or isinstance(rewards, np.ndarray):
            if self.reward_aggregation == "sum":
                return np.sum(rewards)  # Summing the rewards
            elif self.reward_aggregation == "mean":
                return np.mean(rewards)  # Averaging the rewards
            elif self.reward_aggregation == "min":
                return np.min(rewards)  # Taking the minimum reward
            elif self.reward_aggregation == "max":
                return np.max(rewards)  # Taking the maximum reward
            else:
                raise ValueError(
                    f"Unknown reward aggregation method: {self.reward_aggregation}"
                )
        else:
            return rewards  # If it's a single value, return as is


# Done Wrapper to handle aggregation of multi-agent done flags
class DoneWrapper(gym.Wrapper):
    def __init__(self, env, done_aggregation="any"):
        super().__init__(env)
        self.done_aggregation = done_aggregation

    def step(self, action):
        obs, rewards, terminated, truncated, infos = self.env.step(action)

        # Aggregate terminated and truncated flags
        if isinstance(terminated, np.ndarray):
            if self.done_aggregation == "any":
                terminated = np.any(terminated)  # Episode ends if any agent is done
            elif self.done_aggregation == "all":
                terminated = np.all(terminated)  # Episode ends if all agents are done
            else:
                raise ValueError(
                    f"Unknown done aggregation method: {self.done_aggregation}"
                )

        if isinstance(truncated, np.ndarray):
            if self.done_aggregation == "any":
                truncated = np.any(truncated)  # Episode ends if any agent is truncated
            elif self.done_aggregation == "all":
                truncated = np.all(
                    truncated
                )  # Episode ends if all agents are truncated
            else:
                raise ValueError(
                    f"Unknown done aggregation method: {self.done_aggregation}"
                )

        return obs, rewards, terminated, truncated, infos


# Use VMAS environment
def main(
    scenario: str = "flocking",
    seed: int = 0,
    num_envs: int = 10,
    total_timesteps_training: int = 10000,
):
    # Create VMAS environment
    vmas_env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        seed=seed,
        wrapper=Wrapper.GYMNASIUM_VEC,
        terminated_truncated=True,  # needed if you use GYMNASIUM_VEC
    )

    # Apply observation wrapper to flat observation space
    wrapped_env = ObservationWrapper(vmas_env)

    # Apply action wrapper to flat action space
    wrapped_env = ActionWrapper(wrapped_env)

    # Apply the RewardWrapper to aggregate multi-agent rewards
    wrapped_env = RewardWrapper(wrapped_env, reward_aggregation="sum")

    # Apply the DoneWrapper to handle terminated and truncated flags
    wrapped_env = DoneWrapper(wrapped_env, done_aggregation="any")

    # Initialize SB3 model
    model = PPO("MlpPolicy", wrapped_env, verbose=1, seed=seed)

    # Train the model
    model.learn(total_timesteps=total_timesteps_training, progress_bar=True)


if __name__ == "__main__":
    main()
