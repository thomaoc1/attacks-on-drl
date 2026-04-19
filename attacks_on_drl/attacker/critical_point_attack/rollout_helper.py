from typing import Protocol

import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class RolloutHelper(Protocol):
    def collect_baseline_observation(self, observation: VecEnvObs) -> torch.Tensor: ...
    def collect_all_rollout_observations(self, observation: VecEnvObs) -> torch.Tensor: ...
    def get_action_sequence(self, idx: int) -> tuple[int, ...]: ...
