"""Neon Pygame front end: python -m tron_ai.play."""
from __future__ import annotations

import argparse
import math

from tron_ai.engine import Direction, EndReason, VECTORS, TronConfig
from tron_ai.opponents import OPPONENTS
from tron_ai.session import GameSession, Phase

WIDTH, HEIGHT = 1200, 840
BG = (7, 13, 24)
PANEL = (16, 26, 42)
GRID = (17, 33, 50)
EDGE = (41, 64, 85)
WHITE = (236, 247, 255)
MUTED = (150, 175, 195)
COLORS = ((42, 225, 255), (255, 153, 53))
REASONS = {EndReason.HEAD_ON: 'Both entered the same cell',
           EndReason.HEAD_SWAP: 'Both entered the other head',
           EndReason.DOUBLE_CRASH: 'Both cycles crashed',
           EndReason.STEP_LIMIT: 'Turn limit reached',
           EndReason.PLAYER_CRASH: 'One cycle crashed'}


class Renderer:
    def __init__(self, pg):
        self.pg = pg
        self.canvas = pg.Surface((WIDTH, HEIGHT))
        self.fonts = {size: pg.font.Font(None, size) for size in (20, 22, 24, 26, 28, 32, 42, 54, 76, 100)}

    def text(self, string, x, y, size=24, color=WHITE, center=False):
        img = self.fonts[size].render(str(string), True, color)
        rect = img.get_rect(center=(x, y)) if center else img.get_rect(topleft=(x, y))
        self.canvas.blit(img, rect)

    def rect(self, color, bounds, radius=0, width=0):
        self.pg.draw.rect(self.canvas, color, bounds, width, border_radius=radius)

    def draw(self, session):
        pg, s = self.pg, self.canvas
        state, cfg = session.env.state, session.env.config
        s.fill(BG)
        self.text('TRON / LIGHT CYCLES', 32, 22, 42)
        self.text('HUMAN VS AI OPPONENT', 34, 68, 24, MUTED)
        self.text(f'{session.speed} moves/sec   |   TURN {state.step_count:03d}', 843, 37, 24, MUTED)
        # Left: arena. Right: opponent selection and scoreboard.
        cell = min(780 // cfg.width, 624 // cfg.height)
        bw, bh = cell * cfg.width, cell * cfg.height
        ox, oy = 32 + (780 - bw) // 2, 110 + (624 - bh) // 2
        self.rect((9, 19, 32), (ox, oy, bw, bh))
        for x in range(cfg.width + 1):
            pg.draw.line(s, GRID, (ox + x * cell, oy), (ox + x * cell, oy + bh))
        for y in range(cfg.height + 1):
            pg.draw.line(s, GRID, (ox, oy + y * cell), (ox + bw, oy + y * cell))
        self.rect(EDGE, (ox - 2, oy - 2, bw + 4, bh + 4), width=2)
        for p in (0, 1):
            color = COLORS[p]
            for x, y in state.trails[p]:
                px, py = ox + x * cell, oy + y * cell
                margin = max(1, cell // 10)
                self.rect(tuple(c // 4 for c in color), (px + 1, py + 1, cell - 2, cell - 2), 3)
                self.rect(tuple(int(c * .78) for c in color),
                          (px + margin, py + margin, cell - 2 * margin, cell - 2 * margin), 2)
            x, y = state.heads[p]
            px, py = ox + x * cell, oy + y * cell
            self.rect(color, (px + 1, py + 1, cell - 2, cell - 2), 4)
            cx, cy = px + cell / 2, py + cell / 2
            dx, dy = VECTORS[int(state.directions[p])]
            tip = (cx + dx * cell * .36, cy + dy * cell * .36)
            left = (cx - dx * cell * .2 - dy * cell * .24, cy - dy * cell * .2 + dx * cell * .24)
            right = (cx - dx * cell * .2 + dy * cell * .24, cy - dy * cell * .2 - dx * cell * .24)
            pg.draw.polygon(s, BG, [tip, left, right])
        # Mark crash location, clamping wall impacts to the edge of the board.
        if session.phase == Phase.FINISHED and session.last_result:
            for p, collision in enumerate(session.last_result.collisions):
                if collision is None:
                    continue
                x, y = session.last_result.attempted_heads[p]
                cx = max(ox, min(ox + bw, ox + x * cell + cell // 2))
                cy = max(oy, min(oy + bh, oy + y * cell + cell // 2))
                r = max(5, cell // 2)
                pg.draw.circle(s, WHITE, (cx, cy), r, 2)
                pg.draw.line(s, WHITE, (cx - r, cy - r), (cx + r, cy + r), 2)
                pg.draw.line(s, WHITE, (cx - r, cy + r), (cx + r, cy - r), 2)
        self.rect(PANEL, (836, 110, 332, 624), 12)
        self.text('SELECT OPPONENT', 853, 130, 24, MUTED)
        for i, (key, name) in enumerate(session.opponents.items()):
            y = 164 + i * (35 if len(session.opponents)>5 else 43 if len(session.opponents)>4 else 51)
            active = key == session.opponent
            self.rect((28, 48, 67) if active else BG, (850, y, 304, 33 if len(session.opponents)>5 else 40), 7)
            if active:
                self.rect(COLORS[1], (850, y, 4, 33 if len(session.opponents)>5 else 40), 2)
            self.text(str(i + 1), 862, y + 5, 26, COLORS[1] if active else MUTED)
            self.text(name, 891, y + 6, 24, WHITE if active else MUTED)
        shift = max(0, 164+35*len(session.opponents)-375) if len(session.opponents)>5 else 0
        self.text('ROUND SCORE', 853, 388+shift, 24, MUTED)
        wins, losses, draws = session.scores[session.opponent]
        for i, (name, value, color) in enumerate((('YOU', wins, COLORS[0]), ('AI', losses, COLORS[1]), ('DRAW', draws, MUTED))):
            x = 904 + 97 * i
            self.text(name, x, 429+shift, 24, color, True)
            self.text(value, x, 465+shift, 54, color, True)
        pg.draw.line(s, EDGE, (853, 505+shift), (1151, 505+shift))
        self.text('CONTROLS', 853, 524+shift, 24, MUTED)
        for i, line in enumerate(('Arrows / WASD   Steer', 'SPACE   Start / pause / resume', f'1-{len(session.opponents)}   Select before a round', '+ / -   Change speed', 'F11   Fullscreen', 'ESC   Quit')):
            self.text(line, 853, 550 + shift + 22 * i, 22)
        self.text('CYAN: YOU', 32, 758, 28, COLORS[0])
        self.text('ORANGE: AI', 208, 758, 28, COLORS[1])
        self.text('Avoid all trails and walls. No U-turns. Last cycle standing wins.', 32, 793, 24, MUTED)
        if session.phase != Phase.PLAYING:
            cx, cy = ox + bw // 2, oy + bh // 2
            if session.phase == Phase.READY:
                headline, detail, hint, color = 'READY?', f'Choose opponent 1-{len(session.opponents)}', 'Press SPACE to start', WHITE
            elif session.phase == Phase.COUNTDOWN:
                headline, detail, hint, color = str(max(1, math.ceil(session.countdown))), 'Get ready', 'Arrow keys or WASD to steer', COLORS[0]
            elif session.phase == Phase.PAUSED:
                headline, detail, hint, color = 'PAUSED', 'Round suspended', 'Press SPACE to resume', WHITE
            else:
                headline = 'YOU WIN!' if state.winner == 0 else 'YOU LOSE' if state.winner == 1 else 'DRAW'
                color = COLORS[0] if state.winner == 0 else COLORS[1] if state.winner == 1 else WHITE
                detail = REASONS[state.end_reason]
                if state.end_reason == EndReason.PLAYER_CRASH and session.last_result:
                    loser = 1 - state.winner
                    collision = session.last_result.collisions[loser]
                    detail = ('AI' if loser == 1 else 'You') + ' hit ' + collision.value.replace('_', ' ')
                hint = f'SPACE: next round  /  1-{len(session.opponents)}: switch opponent'
            # Translucent overlay leaves the arena visible after the round.
            overlay = pg.Surface((min(bw - 12, 560), 188), pg.SRCALPHA)
            overlay.fill((5, 12, 23, 230))
            s.blit(overlay, overlay.get_rect(center=(cx, cy)))
            self.text(headline, cx, cy - 48, 76, color, True)
            self.text(detail, cx, cy + 13, 28, WHITE, True)
            self.text(hint, cx, cy + 53, 24, MUTED, True)
        return s


def load_opponents(paths):
    """Load at most one checkpoint per supported algorithm, same board required."""
    from tron_ai.training.checkpoints import load
    from tron_ai.training.core import ObservationPolicy
    from tron_ai.train import policy_factory
    extras={}; board=None
    for path in paths:
        weights,meta=load(path)
        algorithm=meta['algorithm']
        if algorithm not in ('dqn','ppo','evolution') or algorithm in extras:
            raise ValueError('provide at most one checkpoint per DQN, PPO or evolution algorithm')
        candidate=TronConfig(**meta['board'])
        if board is not None and candidate!=board:
            raise ValueError('all checkpoints must use the same board configuration')
        board=candidate
        load(path,config=board,algorithm=algorithm)
        factory=policy_factory(algorithm)
        # Validate model shapes immediately, then bind the individual checkpoint.
        factory(weights,board,0)
        extras[algorithm]=(('Neuroevolution' if algorithm=='evolution' else algorithm.upper()+' checkpoint'),lambda seed,f=factory,w=weights,b=board: ObservationPolicy(f(w,b,seed)))
    return {k:extras[k] for k in ('dqn','ppo','evolution') if k in extras},board


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponent', choices=[*OPPONENTS,'dqn','ppo','evolution'], default=None)
    parser.add_argument('--speed', type=int, choices=range(3, 17), default=8, metavar='3..16')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Render a few frames and exit (works with dummy SDL driver).')
    parser.add_argument('--checkpoint', action='append', default=[], help='load a DQN, PPO or evolution NPZ policy; repeat to load all three')
    args = parser.parse_args(argv)
    try:
        extras,board=load_opponents(args.checkpoint)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    opponent=args.opponent or (next(iter(extras)) if extras else 'space')
    if opponent not in OPPONENTS and opponent not in extras:
        parser.error(f'--opponent {opponent} requires a matching --checkpoint')
    try:
        import pygame as pg
    except ImportError:
        parser.exit(1, 'Pygame is not installed. Run: pip install -e ".[play,dev]"\n')
    pg.display.init()
    pg.font.init()
    try:
        fullscreen = args.fullscreen
        window = pg.display.set_mode((0, 0) if fullscreen else (WIDTH, HEIGHT), pg.FULLSCREEN if fullscreen else pg.RESIZABLE)
        pg.display.set_caption('Tron Light Cycles | Human vs AI')
        renderer = Renderer(pg)
        session = GameSession(config=board, seed=args.seed, opponent=opponent, speed=args.speed, extra_opponents=extras)
        clock = pg.time.Clock()
        headings = {pg.K_UP: Direction.NORTH, pg.K_w: Direction.NORTH,
                    pg.K_RIGHT: Direction.EAST, pg.K_d: Direction.EAST,
                    pg.K_DOWN: Direction.SOUTH, pg.K_s: Direction.SOUTH,
                    pg.K_LEFT: Direction.WEST, pg.K_a: Direction.WEST}
        # Smoke mode exercises all screens without requiring user input.
        if args.smoke_test:
            renderer.draw(session)
            session.start_or_pause()
            renderer.draw(session)
            session.update(3.0)
            renderer.draw(session)
            session.start_or_pause()
            renderer.draw(session)
            session.start_or_pause()
            session.update(3.0)
            while session.phase == Phase.PLAYING:
                session.step()
            renderer.draw(session)
            pg.display.flip()
            print('Pygame smoke test passed (ready/countdown/playing/paused/finished).')
            return
        running = True
        while running:
            dt = clock.tick(60) / 1000
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    key = event.key
                    if key == pg.K_ESCAPE:
                        running = False
                    elif key == pg.K_F11:
                        fullscreen = not fullscreen
                        window = pg.display.set_mode((0, 0) if fullscreen else (WIDTH, HEIGHT), pg.FULLSCREEN if fullscreen else pg.RESIZABLE)
                    elif key == pg.K_SPACE:
                        session.start_or_pause()
                        dt = 0.0
                    elif key in headings:
                        session.steer(headings[key])
                    elif pg.K_1 <= key < pg.K_1 + len(session.opponents):
                        session.select_opponent(list(session.opponents)[key - pg.K_1])
                    elif key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                        session.change_speed(1)
                    elif key in (pg.K_MINUS, pg.K_KP_MINUS):
                        session.change_speed(-1)
                elif event.type == pg.WINDOWFOCUSLOST and session.phase == Phase.PLAYING:
                    session.start_or_pause()
            if not running:
                break
            session.update(dt)
            frame = renderer.draw(session)
            # Letterbox into the current display, maintaining square arena cells.
            window = pg.display.get_surface()
            w, h = window.get_size()
            scale = min(w / WIDTH, h / HEIGHT)
            size = (max(1, round(WIDTH * scale)), max(1, round(HEIGHT * scale)))
            window.fill(BG)
            window.blit(pg.transform.smoothscale(frame, size) if size != (WIDTH, HEIGHT) else frame,
                        ((w - size[0]) // 2, (h - size[1]) // 2))
            pg.display.flip()
    finally:
        pg.quit()


if __name__ == '__main__':
    main()
