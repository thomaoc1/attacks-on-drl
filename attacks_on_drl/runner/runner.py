from dataclasses import dataclass
from stable_baselines3.common.vec_env import VecEnv
import numpy as np
from tqdm import trange

from attacks_on_drl.attacker.attacker import BaseAttacker
from attacks_on_drl.victim.victim import BaseVictim


@dataclass
class AttackResults:
    average_reward: float
    std_reward: float
    average_n_attacks: float
    std_n_attacks: float


class AttackRunner:
    def __init__(
        self,
        env: VecEnv,
        attacker: BaseAttacker,
        victim: BaseVictim,
        episode_max_frames: int | float,
    ):
        self.env = env
        self.victim = victim
        self.episode_max_frames = episode_max_frames
        self.attacker = attacker
        
        self.pbar = None
        self.current_run_n_episodes = 0
        self._reset_current_episode_stats()
        
    def _reset_current_episode_stats(self):
        self.current_episode_reward = np.zeros(self.env.num_envs)
        self.current_episode_total_steps = 0
        
    def _step_env(self, action):
        observation, reward, done, _ = self.env.step(action)
        self.current_episode_reward += reward
        self.current_episode_total_steps += 1
        return observation, done

    def run(self, n_episodes: int) -> AttackResults:
        all_episode_rewards = np.zeros(n_episodes)
        all_presence = np.zeros(n_episodes)
        
        pbar = trange(n_episodes, desc="Running episodes")
        for episode in pbar:
            observation = self.env.reset()
            is_done = False
            n_frames = 0
            while not is_done and n_frames < self.episode_max_frames:
                attacker_observation, is_attacked = self.attacker.step(observation)
                if is_attacked:
                    all_presence[episode] += 1
                action = self.victim.choose_action(attacker_observation)
                observation, is_done = self._step_env(action)
                n_frames += 1
                
            all_episode_rewards[episode] = self.current_episode_reward
            rolling_mean_reward = all_episode_rewards[: episode + 1].mean()
            rolling_mean_presence = all_presence[: episode + 1].mean()
            
            pbar.set_description(
                f"Ep {episode + 1}/{n_episodes} | RollingAvgReward: {rolling_mean_reward:.2f} | RollingAvgAttacks: {rolling_mean_presence:.2f}"
            )
            self._reset_current_episode_stats()
            
        return AttackResults(
            average_reward=all_episode_rewards.mean(),
            std_reward=all_episode_rewards.std(),
            average_n_attacks=all_presence.mean(),
            std_n_attacks=all_presence.std(),
        )
