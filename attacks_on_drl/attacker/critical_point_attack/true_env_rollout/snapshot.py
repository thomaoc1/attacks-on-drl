from copy import deepcopy
from contextlib import contextmanager
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


def _safe_deepcopy(v):
    try:
        return deepcopy(v)
    except (TypeError, AttributeError):
        # Non-copyable objects (mappingproxy, C extensions, etc.)
        # are configuration, not mutable state — safe to share the reference
        return v


def _iter_chain(env):
    """
    Yields (kind, wrapper) for each layer in the correct order.

    SB3 VecEnvWrappers use .venv to point inward.
    DummyVecEnv is the base vec layer, its inner envs use .env.
    """
    current = env
    # SB3 VecEnv wrapper layers
    while isinstance(current, VecEnvWrapper):
        yield "vec", current
        current = current.venv

    # Base DummyVecEnv
    assert isinstance(current, DummyVecEnv), f"Expected DummyVecEnv, got {type(current)}"
    yield "dummy", current

    # Inner gym wrapper chain
    inner = current.envs[0]
    while inner is not None:
        yield "gym", inner
        inner = getattr(inner, "env", None)


def save_env_snapshot(env) -> list:
    snapshots = []
    for kind, wrapper in _iter_chain(env):
        if hasattr(wrapper, "ale"):
            snapshots.append((kind, "ale", wrapper, wrapper.ale.cloneState()))  # type: ignore[attr-defined]
        elif kind == "dummy":
            state = {k: _safe_deepcopy(v) for k, v in wrapper.__dict__.items() if k != "envs"}
            snapshots.append((kind, "generic", wrapper, state))
        else:
            exclude = {"venv", "env"}
            state = {k: _safe_deepcopy(v) for k, v in wrapper.__dict__.items() if k not in exclude}
            snapshots.append((kind, "generic", wrapper, state))
    return snapshots


def restore_env_snapshot(snapshots: list) -> None:
    for _, variant, wrapper, state in reversed(snapshots):
        if variant == "ale":
            wrapper.ale.restoreState(state)
        else:
            fresh = {k: _safe_deepcopy(v) for k, v in state.items()}
            wrapper.__dict__.update(fresh)


@contextmanager
def env_snapshot(env):
    snapshot = save_env_snapshot(env)
    try:
        yield snapshot
    finally:
        restore_env_snapshot(snapshot)
