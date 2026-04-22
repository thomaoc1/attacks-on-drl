import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from torchattacks import FGSM

from attacks_on_drl.attacker.common.victim_module_wrapper import VictimModuleWrapper
from attacks_on_drl.victim.common import BaseVictim

from .common import BaseAttacker


class ValueFunctionAttacker(BaseAttacker):
    def __init__(self, victim: BaseVictim, attack_threshold: float, eps: float = 8 / 255) -> None:
        super().__init__(victim=victim)
        self.attack_threshold = attack_threshold

        wrapped_victim = VictimModuleWrapper(self.victim)
        self._perturbation_method = FGSM(wrapped_victim, eps=eps)

    def step(self, obs: VecEnvObs) -> tuple[VecEnvObs, bool]:
        value = self.victim.eval_state(obs).item()
        is_attacked = value > self.attack_threshold
        if is_attacked:
            tens_obs = torch.from_numpy(obs)
            actions = torch.tensor(self.victim.choose_action(obs, deterministic=True))
            adversarial_obs = self._perturbation_method(tens_obs, actions).numpy()
        else:
            adversarial_obs = obs

        return adversarial_obs, is_attacked
