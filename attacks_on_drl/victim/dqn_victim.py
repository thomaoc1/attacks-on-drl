from typing import Iterator

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from attacks_on_drl.victim.victim import BaseVictim


class DQNVictim(BaseVictim[DQN]):
    def model_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.q_net.features_extractor.parameters()

    def eval_state(self, observation: VecEnvObs) -> torch.Tensor:
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        tensor_observation = torch.from_numpy(observation)
        tensor_observation = self._ensure_batch(tensor_observation)

        q_values = self.model.q_net(tensor_observation)
        v_value = q_values.max(dim=-1).values

        return v_value.unsqueeze(1)

    def get_action_logits(self, observation: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(observation, torch.Tensor):
            observation = torch.from_numpy(observation)
            
        observation = self._ensure_batch(observation)
        q_values = self.model.q_net(observation)

        return q_values
