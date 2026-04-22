from typing import Iterator

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from .common import BaseVictim


class DQNVictim(BaseVictim[DQN]):
    def model_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.q_net.features_extractor.parameters()

    def eval_state(self, obs: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, tuple) or isinstance(obs, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)

        obs = self._ensure_batch(obs)

        q_values = self.model.q_net(obs)
        v_value = q_values.max(dim=-1).values

        return v_value.unsqueeze(1)

    def get_action_logits(self, obs: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, tuple) or isinstance(obs, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)

        obs = self._ensure_batch(obs)
        q_values = self.model.q_net(obs)

        return q_values
