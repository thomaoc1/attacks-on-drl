import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from .common.base_victim import BaseVictim


class ActorCriticVictim(BaseVictim):
    def model_parameters(self):
        return self.model.policy.features_extractor.parameters()

    def eval_state(self, obs: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, tuple) or isinstance(obs, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)

        return self.model.policy.predict_values(obs)

    def get_action_logits(self, obs: VecEnvObs | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, tuple) or isinstance(obs, dict):
            raise NotImplementedError("Tuple and dictionary observations not supported")

        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)

        if self.model.policy.share_features_extractor:
            features = self.model.policy.extract_features(obs, self.model.policy.features_extractor)
        else:
            features = self.model.policy.extract_features(obs, self.model.policy.pi_features_extractor)

        latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
        logits = self.model.policy.action_net(latent_pi)
        return logits
