from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from torchattacks import FGSM
from torchattacks.attack import torch

from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.victim.common import BaseVictim

from .common import BaseAttacker


class FGSMAttacker(BaseAttacker):
    def __init__(self, victim: BaseVictim, eps: float = 8 / 255) -> None:
        super().__init__(victim=victim)
        wrapped_victim = VictimModuleWrapper(self.victim)
        self._perturbation_method = FGSM(wrapped_victim, eps=eps)

    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        actions = torch.tensor(self.victim.choose_action(observation, deterministic=True))
        return self._perturbation_method(torch.from_numpy(observation), actions), True
