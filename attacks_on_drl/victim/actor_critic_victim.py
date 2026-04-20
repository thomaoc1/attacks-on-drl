import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from .common.base_victim import BaseVictim


class ActorCriticVictim(BaseVictim):
    def model_parameters(self):
        return self.model.policy.features_extractor.parameters()

    def eval_state(self, observation: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(observation, torch.Tensor):
            observation = torch.from_numpy(observation)

        return self.model.policy.predict_values(observation)

    def get_action_logits(self, observation: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(observation, tuple) or isinstance(observation, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(observation, torch.Tensor):
            observation = torch.from_numpy(observation)

        if self.model.policy.share_features_extractor:
            features = self.model.policy.extract_features(observation, self.model.policy.features_extractor)
        else:
            features = self.model.policy.extract_features(observation, self.model.policy.pi_features_extractor)

        latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
        logits = self.model.policy.action_net(latent_pi)
        return logits
