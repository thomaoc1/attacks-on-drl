from typing import cast

import torch
import torchattacks
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from attacks_on_drl.attacker.attacker import BaseAttacker
from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.attacker.critical_point_attack.divergence import AtariDivergenceFunction
from attacks_on_drl.attacker.critical_point_attack.true_env_rollout.ale_types import ALEEnvProtocol
from attacks_on_drl.attacker.critical_point_attack.true_env_rollout.ram_rollout_helper import RamRolloutHelper
from attacks_on_drl.attacker.critical_point_attack.true_env_rollout.wrappers import ScaledAtariVecWrapper
from attacks_on_drl.victim.base_victim import BaseVictim


class CriticalPointAttack(BaseAttacker):
    def __init__(
        self,
        env: ScaledAtariVecWrapper,
        victim: BaseVictim,
        attack_threshold: float,
        cw_kwargs: dict | None = None,
    ) -> None:
        super().__init__(victim=victim)

        self.rollout_helper = RamRolloutHelper(env, victim, 2, 2)
        env0 = cast(ALEEnvProtocol, env.envs[0].unwrapped)
        self.divergence_function = AtariDivergenceFunction(env0.spec.id)
        self.attack_threshold = attack_threshold
        wrapped_victim = VictimModuleWrapper(self.victim)

        if not cw_kwargs:
            cw_kwargs = dict()

        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label(quiet=True)

        self.current_attack_action_seq = None
        self.current_attack_action_seq_idx = 0

    def _attack(self, observation: VecEnvObs) -> VecEnvObs:
        assert self.current_attack_action_seq is not None, "Current action sequence is None."

        target_action = torch.tensor(self.current_attack_action_seq[self.current_attack_action_seq_idx]).unsqueeze(0)
        tens_observation = torch.from_numpy(observation)
        adversarial_observation = self._perturbation_method(tens_observation, target_action).numpy()

        self.current_attack_action_seq_idx += 1
        if self.current_attack_action_seq_idx == len(self.current_attack_action_seq):
            self.current_attack_action_seq = None
            self.current_attack_action_seq_idx = 0

        return adversarial_observation

    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        if self.current_attack_action_seq is not None:
            return self._attack(observation), True

        baseline_value = self.divergence_function(self.rollout_helper.collect_baseline_observation(observation)).item()

        all_final_ram_states = self.rollout_helper.collect_all_final_ram(observation)
        best_attack_value, best_attack_value_idx = torch.max(self.divergence_function(all_final_ram_states), dim=0)

        if abs(best_attack_value.item() - baseline_value) > self.attack_threshold:
            self.current_attack_action_seq = self.rollout_helper.get_action_sequence(int(best_attack_value_idx.item()))
            return self._attack(observation), True

        return observation, False
