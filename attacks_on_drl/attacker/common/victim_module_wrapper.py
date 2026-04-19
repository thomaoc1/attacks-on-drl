from typing import Iterator

import torch

from attacks_on_drl.victim.common.base_victim import BaseVictim


class VictimModuleWrapper(torch.nn.Module):
    def __init__(self, victim_agent: BaseVictim):
        super().__init__()
        self.agent = victim_agent

    def forward(self, obs):
        return self.agent.get_action_logits(obs)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return self.agent.model_parameters()
