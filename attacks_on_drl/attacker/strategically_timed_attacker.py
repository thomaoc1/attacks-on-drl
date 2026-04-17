import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
import torchattacks
from attacks_on_drl.attacker.attacker import BaseAttacker
from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.victim.victim import BaseVictim


class StrategicallyTimedAttacker(BaseAttacker):
    def __init__(self, victim: BaseVictim, attack_threshold: float, cw_kwargs: dict | None = None) -> None:
        super().__init__(victim=victim)
        self.attack_threshold = attack_threshold
        
        wrapped_victim = VictimModuleWrapper(self.victim)
        if not cw_kwargs:
            cw_kwargs = dict()
        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label()
        
    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        policy_logits = self.victim.get_action_logits(observation)
        softmax_logits = F.softmax(policy_logits, dim=1)
        attack_indicator = softmax_logits.max() - softmax_logits.min()
    
        if attack_indicator > self.attack_threshold:
            tens_observation = torch.from_numpy(observation)
            adversarial_observation = self._perturbation_method(tens_observation, softmax_logits.argmin(dim=1)).numpy()
            return adversarial_observation, True
            
        return observation, False