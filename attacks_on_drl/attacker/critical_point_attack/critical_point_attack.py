from typing import Callable

import torch
import torchattacks
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from attacks_on_drl.attacker.common import BaseAttacker, VictimModuleWrapper
from attacks_on_drl.victim.common import BaseVictim
from .rollout_helper import RolloutHelper


class CriticalPointAttack(BaseAttacker):
    def __init__(
        self,
        victim: BaseVictim,
        rollout_helper: RolloutHelper,
        divergence_function: Callable[[torch.Tensor], torch.Tensor],
        attack_threshold: float,
        cw_kwargs: dict | None = None,
    ) -> None:
        super().__init__(victim=victim)

        self.rollout_helper = rollout_helper
        self.divergence_function = divergence_function
        self.attack_threshold = attack_threshold
        wrapped_victim = VictimModuleWrapper(self.victim)

        if not cw_kwargs:
            cw_kwargs = dict()

        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label(quiet=True)

        self.current_attack_action_seq = None
        self.current_attack_action_seq_idx = 0

    def _attack(self, obs: VecEnvObs) -> VecEnvObs:
        assert self.current_attack_action_seq is not None, "Current action sequence is None."

        target_action = torch.tensor(self.current_attack_action_seq[self.current_attack_action_seq_idx]).unsqueeze(0)
        tens_obs = torch.from_numpy(obs)
        adversarial_obs = self._perturbation_method(tens_obs, target_action).numpy()

        self.current_attack_action_seq_idx += 1
        if self.current_attack_action_seq_idx == len(self.current_attack_action_seq):
            self.current_attack_action_seq = None
            self.current_attack_action_seq_idx = 0

        return adversarial_obs

    def step(self, obs: VecEnvObs) -> tuple[VecEnvObs, bool]:
        if self.current_attack_action_seq is not None:
            return self._attack(obs), True

        baseline_value = self.divergence_function(self.rollout_helper.collect_baseline_obs(obs)).item()

        all_final_ram_states = self.rollout_helper.collect_all_rollout_obs(obs)
        best_attack_value, best_attack_value_idx = torch.max(self.divergence_function(all_final_ram_states), dim=0)

        if abs(best_attack_value.item() - baseline_value) > self.attack_threshold:
            self.current_attack_action_seq = self.rollout_helper.get_action_sequence(int(best_attack_value_idx.item()))
            return self._attack(obs), True

        return obs, False
