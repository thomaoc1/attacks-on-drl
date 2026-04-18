import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class ScaledAtariVecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        observation_space = venv.observation_space
        self.observation_space = type(observation_space)(
            low=0.0, high=1.0, shape=observation_space.shape, dtype=np.float32
        )

    def reset(self):
        observation = self.venv.reset()
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        return observation.astype(np.float32) / 255.0

    def step_wait(self):
        observation, rewards, dones, infos = self.venv.step_wait()
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        return observation.astype(np.float32) / 255.0, rewards, dones, infos
