import torch
import torch.nn.functional as F
import torchattacks
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.victim.common import BaseVictim

from .common import BaseAttacker


class StrategicallyTimedAttacker(BaseAttacker):
    def __init__(self, victim: BaseVictim, attack_threshold: float, cw_kwargs: dict | None = None) -> None:
        super().__init__(victim=victim)
        self.attack_threshold = attack_threshold

        wrapped_victim = VictimModuleWrapper(self.victim)
        if not cw_kwargs:
            cw_kwargs = dict()
        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label()

    def step(self, obs: VecEnvObs) -> tuple[VecEnvObs, bool]:
        policy_logits = self.victim.get_action_logits(obs)
        softmax_logits = F.softmax(policy_logits, dim=1)
        attack_indicator = softmax_logits.max() - softmax_logits.min()

        if attack_indicator > self.attack_threshold:
            tens_obs = torch.from_numpy(obs)
            adversarial_obs = self._perturbation_method(tens_obs, softmax_logits.argmin(dim=1)).numpy()
            return adversarial_obs, True

        return obs, False
