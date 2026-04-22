from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from torchattacks import FGSM
from torchattacks.attack import torch

from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.victim.common import BaseVictim

from .common import BaseAttacker


class FGSMEveryNAttacker(BaseAttacker):
    def __init__(self, victim: BaseVictim, attack_every_n_steps: int, eps: float = 8 / 255) -> None:
        super().__init__(victim=victim)
        wrapped_victim = VictimModuleWrapper(self.victim)
        self._perturbation_method = FGSM(wrapped_victim, eps=eps)
        self.attack_every_n_steps = attack_every_n_steps
        self.current_step = 0
        self.current_perturbation = None

    def step(self, obs: VecEnvObs) -> tuple[VecEnvObs, bool]:
        tens_obs = torch.from_numpy(obs)
        if self.current_step % self.attack_every_n_steps == 0:
            actions = torch.tensor(self.victim.choose_action(obs, deterministic=True))
            self.current_perturbation = self._perturbation_method(tens_obs, actions)

        self.current_step += 1
        assert self.current_perturbation is not None, f"No perturbation computed on step {self.current_step}."
        return self.current_perturbation, True
