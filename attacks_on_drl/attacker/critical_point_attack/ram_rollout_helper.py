from itertools import product
from typing import cast

from gymnasium.spaces import Discrete
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from attacks_on_drl.attacker.critical_point_attack.ale_types import ALEEnvProtocol
from attacks_on_drl.victim.victim import BaseVictim


class RamRolloutHelper:
    def __init__(
        self, env: DummyVecEnv, victim: BaseVictim, action_enumeration_len: int, baseline_state_distance: int
    ) -> None:
        assert action_enumeration_len > 0, "Action enumeration length must be greater than 0."
        assert action_enumeration_len <= baseline_state_distance, (
            "Action enumeration length must be smaller than or equal to baseline state distance."
        )

        self.env = env
        self.env0 = cast(ALEEnvProtocol, self.env.envs[0].unwrapped)
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

    def collect_all_final_ram(self, observation: VecEnvObs) -> torch.Tensor:
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        real_states = []
        original_state = self.env0.clone_state()

        for action_sequence in self.action_enumeration_np:
            for action in action_sequence:
                current_observation, _, _, _ = self.env.step(action)

            for _ in range(self.baseline_state_distance - self.action_enumeration_len):
                action = self.victim.choose_action(current_observation, deterministic=True)
                current_observation, _, _, _ = self.env.step(action)

            real_states.append(torch.from_numpy(self.env0.ale.getRAM()))
            self.env0.restore_state(original_state)

        return torch.stack(real_states)

    def collect_baseline_observation(self, observation: VecEnvObs) -> torch.Tensor:
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        original_state = self.env0.clone_state()
        for _ in range(self.baseline_state_distance):
            action = self.victim.choose_action(observation, deterministic=True)
            observation, _, _, _ = self.env.step(action)

        final_ram_state = self.env0.ale.getRAM()
        self.env0.restore_state(original_state)

        final_ram_state = torch.from_numpy(final_ram_state).unsqueeze(0)
        return final_ram_state
