from itertools import product

import numpy as np
import torch
from gymnasium.spaces import Discrete
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from attacks_on_drl.attacker.critical_point_attack.rollout_helper import RolloutHelper
from attacks_on_drl.victim.common import BaseVictim

from .snapshot import env_snapshot, restore_env_snapshot


class RamRolloutHelper(RolloutHelper):
    def __init__(
        self, env: VecEnv, victim: BaseVictim, action_enumeration_len: int, baseline_state_distance: int
    ) -> None:
        assert action_enumeration_len > 0, "Action enumeration length must be greater than 0."
        assert action_enumeration_len <= baseline_state_distance, (
            "Action enumeration length must be smaller than or equal to baseline state distance."
        )

        self.env = env
        self.env_ale = self.env.get_attr("ale")[0]
        if self.env_ale is None:
            raise ValueError("Environment must be from ALE.")

        self.victim = victim
        self.action_enumeration_len = action_enumeration_len
        self.baseline_state_distance = baseline_state_distance

        if not isinstance(self.env.action_space, Discrete):
            raise TypeError("Expected Discrete action space")

        n_actions = self.env.action_space.n
        actions = [i for i in range(n_actions)]
        self.action_enumeration = list(product(actions, repeat=self.action_enumeration_len))
        self.action_enumeration_np = [
            tuple(np.array([action], dtype=np.uint8) for action in action_sequence)
            for action_sequence in list(product(actions, repeat=self.action_enumeration_len))
        ]

    def get_action_sequence(self, maximising_idx: int) -> tuple[int, ...]:
        return self.action_enumeration[maximising_idx]

    def collect_all_rollout_obs(self, obs: VecEnvObs) -> torch.Tensor:
        if isinstance(obs, (tuple, dict)):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        real_states = []
        with env_snapshot(self.env) as snapshot:
            for action_sequence in self.action_enumeration_np:
                current_obs = obs

                done = False
                for action in action_sequence:
                    if done:
                        break
                    current_obs, _, done, _ = self.env.step(action)

                for _ in range(self.baseline_state_distance - self.action_enumeration_len):
                    if done:
                        break
                    action = self.victim.choose_action(current_obs, deterministic=True)
                    current_obs, _, done, _ = self.env.step(action)

                real_states.append(torch.from_numpy(self.env_ale.getRAM()))
                restore_env_snapshot(snapshot)

        return torch.stack(real_states)

    def collect_baseline_obs(self, obs: VecEnvObs) -> torch.Tensor:
        if isinstance(obs, (tuple, dict)):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        with env_snapshot(self.env):
            current_obs = obs
            for _ in range(self.baseline_state_distance):
                action = self.victim.choose_action(current_obs, deterministic=True)
                current_obs, _, _, _ = self.env.step(action)

            final_ram = torch.from_numpy(self.env_ale.getRAM()).unsqueeze(0)

        return final_ram
