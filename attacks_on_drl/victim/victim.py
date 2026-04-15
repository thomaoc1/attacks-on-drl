from abc import abstractmethod
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
import numpy as np


class BaseVictim:
    @abstractmethod
    def choose_action(self, observation: VecEnvObs) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        pass
    
