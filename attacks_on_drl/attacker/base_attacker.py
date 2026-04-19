from abc import ABC, abstractmethod

from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from attacks_on_drl.victim.base_victim import BaseVictim


class BaseAttacker(ABC):
    def __init__(self, victim: BaseVictim) -> None:
        super().__init__()
        self.victim = victim

    @abstractmethod
    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        pass
