"""No-Pygame smoke demo: python -m tron_ai --games 5 --seed 2026"""

from __future__ import annotations

import argparse
import random

from tron_ai import Action, TronEnv


def render_ascii(env: TronEnv) -> str:
    state, cfg = env.state, env.config
    grid = [[" " for _ in range(cfg.width)] for _ in range(cfg.height)]
    for player, marker in ((0, "."), (1, ":")):
        for x, y in state.trails[player]:
            grid[y][x] = marker
    for player, marker in ((0, "A"), (1, "B")):
        x, y = state.heads[player]
        grid[y][x] = marker
    border = "+" + "-" * cfg.width + "+"
    return "\n".join([border, *("|" + "".join(r) + "|" for r in grid), border])


def main() -> None:
    parser = argparse.ArgumentParser(description="Tron engine smoke test (no Pygame)")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--show-board", action="store_true")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    rng = random.Random(args.seed)
    env = TronEnv(seed=args.seed)
    totals = [0, 0, 0]  # p0 wins, p1 wins, draws
    for game in range(args.games):
        env.reset(seed=args.seed + game, mode="random_mirrored")
        while not env.state.done:
            # Policies would examine env.observe(0/1) BEFORE this atomic step.
            actions = (Action(rng.randrange(3)), Action(rng.randrange(3)))
            env.step(actions)
        index = 2 if env.state.winner is None else env.state.winner
        totals[index] += 1
        print(f"Game {game + 1}: winner={env.state.winner}, "
              f"reason={env.state.end_reason.value}, "
              f"turns={env.state.step_count}")
        if args.show_board:
            print(render_ascii(env))
    print(f"Total: P0 wins={totals[0]}, P1 wins={totals[1]}, draws={totals[2]}")


if __name__ == "__main__":
    main()
