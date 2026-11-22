# Tron Light Cycles — Offline Ubuntu Edition

Play Tron against scripted opponents and two trained reinforcement-learning agents, **DQN (Intermediate)** and **PPO (Advanced)**. This repository includes the game source, trained checkpoints, and pre-bundled CPU-only Python packages. **You do not need to run `pip install` on the destination.**

## Quick start (Ubuntu)

This build is intended for **Ubuntu/Linux x86-64 with Python 3.12** and a working graphical desktop.

```bash
# Clone using SSH (requires GitHub SSH access configured on this machine):
git clone git@github.com:HoawenLo/tron_light_cycle.git
cd tron_light_cycle

# Check that the game and its bundled dependencies load:
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./run.sh --smoke-test

# Start the playable game:
./run.sh
```

If SSH access is unavailable and your repository permissions permit HTTPS cloning, use:

```bash
git clone https://github.com/HoawenLo/tron_light_cycle.git
cd tron_light_cycle
```

On first launch, `run.sh` **automatically reassembles and extracts** the bundled libraries. This may take a little longer than later launches. No manual archive extraction, pip command, virtual environment, model download, or training is required.

## Controls

| Key | Action |
| --- | --- |
| Arrow keys or WASD | Steer your light cycle |
| Space | Start / pause / resume; start the next round when finished |
| `1` | Random scripted opponent |
| `2` | Random Safe scripted opponent |
| `3` | Flood-fill scripted opponent |
| `4` | Territory Control scripted opponent |
| `5` | **DQN — Intermediate** |
| `6` | **PPO — Advanced** (selected by default) |
| `+` / `-` | Increase / decrease game speed |
| `F11` | Toggle fullscreen |
| `Esc` | Quit |

Select an opponent before starting a round (or after a round finishes). Avoid walls and light-cycle trails. Your cycle continues moving until you turn; reversing direction is not allowed.

## What is included?

```text
tron_light_cycle/
├── app/src/tron_ai/           # Tron game, scripted opponents and AI code
├── models/
│   ├── dqn_intermediate.npz   # Best retained DQN checkpoint
│   └── ppo_advanced.npz       # Best validated PPO checkpoint
├── vendor.tar.gz.part-aa      # Bundled offline Python dependencies
├── vendor.tar.gz.part-ab
├── vendor.tar.gz.part-ac
├── vendor.tar.gz.part-ad
├── vendor.tar.gz.part-ae
├── vendor.tar.gz.part-af
├── run.sh                     # Launcher; automatically extracts dependencies
└── README.md
```

The archive pieces are **not wheel files**. They contain already-installed versions of NumPy, Pygame, CPU-only PyTorch and their Python dependencies. `run.sh` combines these pieces and extracts them into a local `vendor/` directory on first use. Do not delete any of the six archive pieces from the Git repository.

## Requirements and limitations

- **Linux x86-64 (AMD64)**. This bundle will not run on ARM64 without a separately prepared dependency build.
- **Python 3.12**, available as `python3.12`.
- A graphical desktop session with compatible native Linux libraries, including the system C/C++ runtime and graphics/display libraries used by Pygame and PyTorch.
- **Git** to clone the repository and ordinary `bash`, `tar`, and `cat` command-line tools.
- No NVIDIA GPU, CUDA, pip installation, or internet access for Python dependencies is required. Access to the Git server is still needed for the initial clone.

The bundled wheels include native components targeting a sufficiently recent GNU/Linux runtime (including glibc 2.28 or later); the first run on an untested Ubuntu machine is not guaranteed until the smoke test succeeds.

## Common problems

**`Permission denied` when starting `./run.sh`**

```bash
chmod +x run.sh
./run.sh
```

**`python3.12: command not found`** — Python 3.12 is not available on the destination PATH. This repository bundles Python *packages*, not the Python interpreter itself.

**A Python import error, shared-library error, or graphical-display error** — Confirm the machine is Linux x86-64, has Python 3.12, and has a functioning desktop session. Check the error shown by:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./run.sh --smoke-test
```

If the smoke test succeeds but the real game does not open a window, check that you are running from an Ubuntu desktop session with a working display (rather than a headless SSH shell).

**Archives are missing or incomplete** — Run `ls -lh vendor.tar.gz.part-*` and ensure the six archive parts exist. A failed first extraction may leave a partially populated `vendor/` directory; remove only that generated directory, then rerun:

```bash
rm -rf vendor
./run.sh --smoke-test
```

**`git clone` says permission denied (publickey)** — Configure an SSH key for GitHub on the destination, or use the HTTPS clone URL if permitted. Do not copy your private SSH key from another computer.

## Updating the repository

For new game code or model checkpoints, update the repository from your development computer, then on the destination:

```bash
cd tron_light_cycle
git pull --ff-only
./run.sh
```

If the dependency archive changes in a future release, remove the generated `vendor/` directory before launching again so it gets reconstructed. Otherwise it is reused automatically.

## Development note

The two network models use CPU inference only. **DQN and PPO are learned agents; Territory Control is a separate scripted search-based opponent.** The published models are intended for gameplay demonstrations; you do not need to retrain anything to play.
