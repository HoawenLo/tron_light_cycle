"""Conservative PPO fine-tuning: same actor-critic, smaller policy updates."""
from dataclasses import dataclass
from .advanced_ppo import EnhancedPPOConfig, EnhancedPPOLearner


@dataclass(frozen=True)
class StablePPOConfig(EnhancedPPOConfig):
    lr: float = 0.00003
    lr_end: float = 0.000012
    clip: float = .15
    target_kl: float = .015
    entropy_coef: float = .01
    anneal_steps: int = 400000


class StablePPOLearner(EnhancedPPOLearner):
    """No architecture changes; use established PPO training path."""
