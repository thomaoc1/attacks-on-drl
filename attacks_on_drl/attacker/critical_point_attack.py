import torchattacks
from attacks_on_drl.attacker.attacker import BaseAttacker
from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.victim.victim import BaseVictim


class CriticalPointAttack(BaseAttacker):
    def __init__(self, victim: BaseVictim, attack_threshold: float, cw_kwargs: dict | None = None) -> None:
        super().__init__(victim=victim)
        self.attack_threshold = attack_threshold

        wrapped_victim = VictimModuleWrapper(self.victim)
        if not cw_kwargs:
            cw_kwargs = dict()
        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label()
