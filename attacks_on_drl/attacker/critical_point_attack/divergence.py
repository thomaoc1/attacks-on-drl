import torch

RAM_ANNOTATIONS = {
    "pong": dict(
        player_x=46,
        player_y=51,
        enemy_x=45,
        enemy_y=50,
        ball_x=49,
        ball_y=54,
    ),
    "breakout": dict(
        ball_x=99,
        ball_y=101,
        player_x=72,
        player_y=192,
        blocks_hit_count=77,
        block_bit_map=range(30),
        score=84,
    ),
}


def normalize_env_name(env: str) -> str:
    env = env.lower()

    if "pong" in env:
        return "pong"
    if "breakout" in env:
        return "breakout"

    raise ValueError(f"Unsupported environment: {env}")


class AtariDivergenceFunction:
    def __init__(self, env: str):
        self._env = normalize_env_name(env)
        self._annotations = RAM_ANNOTATIONS[self._env]

        self._drop_conditions = {
            "pong": lambda bx, by: (bx > 191) & (bx < 198),
            "breakout": lambda bx, by: (by > 176) & (by < 187),
        }

        self._player_y_getters = {
            "pong": lambda ram: ram[:, self._annotations["player_y"]],
            "breakout": lambda ram: torch.ones(ram.size(0)) * 192,
        }

    def _is_dropped(self, ball_xs: torch.Tensor, ball_ys: torch.Tensor):
        return self._drop_conditions[self._env](ball_xs, ball_ys)

    def _get_player_y(self, final_ram_state: torch.Tensor):
        return self._player_y_getters[self._env](final_ram_state)

    def __call__(self, final_ram_state: torch.Tensor) -> torch.Tensor:
        player_xs = final_ram_state[:, self._annotations["player_x"]]
        player_ys = self._get_player_y(final_ram_state)
        ball_xs = final_ram_state[:, self._annotations["ball_x"]]
        ball_ys = final_ram_state[:, self._annotations["ball_y"]]
        player_ball_distance = torch.sqrt((player_xs - ball_xs) ** 2 + (player_ys - ball_ys) ** 2)
        is_dropped = self._is_dropped(ball_xs, ball_ys)
        return player_ball_distance * is_dropped
