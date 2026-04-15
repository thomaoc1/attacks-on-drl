from typing import Generic, TypeVar
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
    
    def choose_action(self, observation: VecEnvObs, deterministic: bool):
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")
            
        return self.model.predict(observation, deterministic=deterministic)

    def model_parameters(self):
        raise NotImplementedError("Must be implemented by child class.")

    def eval_state(self, observation: VecEnvObs) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by child class.")

    def get_action_logits(self, observation: VecEnvObs) -> torch.Tensor:
        raise NotImplementedError("Must be implemented by child class.")
