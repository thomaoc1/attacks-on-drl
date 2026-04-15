from stable_baselines3.common.vec_env import VecEnv

from attacks_on_drl.attacker.attacker import BaseAttacker
from attacks_on_drl.victim.victim import BaseVictim

class AttackRunner:
    def __init__(
        self,
        env: VecEnv,
        attacker: BaseAttacker,
        victim: BaseVictim,
        episode_max_frames: int | float,
    ):
        pass