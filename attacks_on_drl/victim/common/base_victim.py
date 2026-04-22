from typing import Generic, Iterator, TypeVar

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

T = TypeVar("T", bound=BaseAlgorithm)


class BaseVictim(Generic[T]):
    def __init__(self, model: T):
        self.model = model

    def _ensure_batch(self, obs: torch.Tensor) -> torch.Tensor:
        if self.model.observation_space.shape and obs.dim() == len(self.model.observation_space.shape):
            obs = obs.unsqueeze(0)
        return obs

    def choose_action(self, obs: VecEnvObs | torch.Tensor, deterministic: bool) -> np.ndarray:
        if isinstance(obs, tuple) or isinstance(obs, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def model_parameters(self) -> Iterator[torch.nn.Parameter]:
        raise NotImplementedError("Must be implemented by child class.")

    def eval_state(self, obs: VecEnvObs | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by child class.")

    def get_action_logits(self, obs: VecEnvObs | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by child class.")
