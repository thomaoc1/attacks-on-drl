from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

class BaseAttacker(ABC):
    @abstractmethod
    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        pass
