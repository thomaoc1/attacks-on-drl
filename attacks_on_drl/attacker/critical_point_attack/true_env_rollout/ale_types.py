from typing import Protocol
import numpy as np


class EnvSpecProtocol(Protocol):
    id: str


class ALEInterface(Protocol):
    def getRAM(self) -> np.ndarray: ...


class ALEEnvProtocol(Protocol):
    ale: ALEInterface
    spec: EnvSpecProtocol

    def clone_state(self) -> object: ...
    def restore_state(self, state: object) -> None: ...
