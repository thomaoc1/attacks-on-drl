from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
import torch
from torchattacks import FGSM

from attacks_on_drl.attacker.base_attacker import BaseAttacker
from attacks_on_drl.attacker.common.victim_module_wrapper import VictimModuleWrapper
from attacks_on_drl.victim.base_victim import BaseVictim


class ValueFunctionAttacker(BaseAttacker):
    def __init__(self, victim: BaseVictim, attack_threshold: float, eps: float = 8 / 255) -> None:
        super().__init__(victim=victim)
        self.attack_threshold = attack_threshold

        wrapped_victim = VictimModuleWrapper(self.victim)
        self._perturbation_method = FGSM(wrapped_victim, eps=eps)

    def _perturb_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return observation

    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        value = self.victim.eval_state(observation).item()
        is_attacked = value > self.attack_threshold
        if is_attacked:
            tens_observation = torch.from_numpy(observation)
            actions = torch.tensor(self.victim.choose_action(observation, deterministic=True))
            adversarial_observation = self._perturbation_method(tens_observation, actions).numpy()
        else:
            adversarial_observation = observation

        return adversarial_observation, is_attacked
