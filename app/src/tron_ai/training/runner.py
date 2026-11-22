"""Time-budgeted, serial training with immutable evaluation snapshots."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import csv, json, os, platform, random, sys, tempfile, time
import numpy as np
from tron_ai.engine import TronConfig
from tron_ai.observations import observation_shape
from .benchmark import VALIDATION_SEED, schedule, evaluate
from .core import Learner, ActorFactory, play_episode, training_opening
from .pool import OpponentPool, STAGES
from .checkpoints import save

@dataclass(frozen=True)
class RunConfig:
    seconds: float = 30
    eval_interval: float = 10
    eval_pairs: int = 2
    seed: int = 2026
    stage: str = 'survival'
    max_episodes: int | None = None
    validation_seed: int = VALIDATION_SEED
    curriculum: tuple[tuple[str,float], ...] = ()
    def __post_init__(self):
        if (not np.isfinite(self.seconds) or not np.isfinite(self.eval_interval) or
                self.seconds<=0 or self.eval_interval<=0 or self.eval_pairs<1 or
                self.stage not in STAGES or (self.max_episodes is not None and self.max_episodes<1)):
            raise ValueError('invalid run configuration')
        if self.curriculum:
            if any(stage not in STAGES or not np.isfinite(f) or f<=0 for stage,f in self.curriculum) or not np.isclose(sum(f for _,f in self.curriculum),1,rtol=0,atol=1e-9):
                raise ValueError('curriculum stages must have positive fractions summing to 1')
            if self.curriculum[0][0]!=self.stage:
                raise ValueError('initial stage must match curriculum')

def stage_index(cfg, elapsed):
    if not cfg.curriculum: return 0
    end=0.
    for i,(_,fraction) in enumerate(cfg.curriculum):
        end+=fraction*cfg.seconds
        if elapsed<end: return i
    return len(cfg.curriculum)-1


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append_jsonl(path, value):
    with path.open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')


def selection_score(metrics: dict, selection: str = 'overall') -> float:
    """Development-set checkpoint score; independent holdout stays untouched.

    strong_balanced = .4*Flood-fill + .4*Territory + .2*weaker-opponent.
    It makes a model that forgets either strong opponent less attractive.
    """
    if selection == 'overall':
        return float(metrics['score'])
    if selection != 'strong_balanced':
        raise ValueError(f'unknown checkpoint-selection strategy: {selection}')
    by = metrics['by_opponent']
    flood, territory = float(by['space']['score']), float(by['territory']['score'])
    return .4*(flood+territory)+.2*min(flood, territory)


def run(learner: Learner, factory: ActorFactory, board: TronConfig, cfg: RunConfig,
        out: Path, *, algorithm_config: dict | None = None, verbose=True,
        resume_state_path: Path | None = None, persist_training_state: bool = False,
        selection: str = 'overall') -> dict:
    if selection not in ('overall', 'strong_balanced'):
        raise ValueError('invalid checkpoint selection method')
    out=Path(out)
    # Fail rather than accidentally mixing runs or overwriting an existing benchmark.
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'output directory not empty: {out}')
    rng=random.Random(cfg.seed)
    pool=OpponentPool(cfg.stage,seed=rng.getrandbits(63))
    open_rng=random.Random(rng.getrandbits(63))
    episode_offset=step_offset=0
    if persist_training_state and not hasattr(learner,'export_state'):
        raise ValueError('learner does not support full-state checkpoints')
    if resume_state_path is not None:
        if not persist_training_state or not hasattr(learner,'import_state'):
            raise ValueError('resume requires full-state checkpoint support')
        import torch
        bundle=torch.load(resume_state_path,map_location='cpu',weights_only=True)
        if bundle['format_version']!=1 or bundle['algorithm']!=learner.algorithm or bundle['board']!=asdict(board):
            raise ValueError('resume checkpoint has incompatible environment or algorithm')
        learner.import_state(bundle['learner'])
        pool.import_state(bundle['pool'],factory)
        if pool.stage != cfg.stage:
            raise ValueError('resume stage must match selected stage')
        open_rng.setstate(bundle['open_rng_state'])
        rng.setstate(bundle['runner_rng_state'])
        episode_offset=int(bundle['cumulative_episodes']);step_offset=int(bundle['cumulative_steps'])
    cases=schedule(cfg.eval_pairs,cfg.validation_seed)
    spec=dict(algorithm=learner.algorithm,board=asdict(board),run=asdict(cfg),
              checkpoint_selection=selection,
              algorithm_config=algorithm_config or {},observation_shape=list(observation_shape(board)),
              action_ids={'left':0,'straight':1,'right':2},reward={'win':1,'draw':0,'loss':-1,'ongoing':0},
              resume_from=str(resume_state_path) if resume_state_path else None,
              python=sys.version,platform=platform.platform(),numpy=np.__version__,
              timing='serial wall-clock episode or full generation execution, including opponents, encoding and updates; evaluation and logging excluded')
    write_json(out/'config.json',spec)
    write_json(out/'evaluation_schedule.json',[asdict(c) for c in cases])
    elapsed=eval_elapsed=checkpoint_elapsed=0.
    episodes=steps=0
    best_score=-1.; best_id=None; best_overall_score=-1.;best_overall_id=None
    eval_index=0
    next_eval=cfg.eval_interval
    current_index=0
    stage_log=[dict(stage=pool.stage,training_seconds=0.,environment_steps=0,episodes=0)]
    write_json(out/'stages.json',stage_log)
    wall_start=time.perf_counter()
    def checkpoint():
        nonlocal eval_elapsed,checkpoint_elapsed,best_score,best_id,best_overall_score,best_overall_id,eval_index
        start=time.perf_counter(); weights=learner.snapshot()
        snapshot_time=time.perf_counter()-start
        start=time.perf_counter(); metrics,rows=evaluate(factory,weights,board,cases)
        duration=time.perf_counter()-start; eval_elapsed+=duration
        start=time.perf_counter()
        name=f'checkpoint_{eval_index:04d}'
        strong_score=selection_score(metrics,'strong_balanced')
        chosen_score=selection_score(metrics,selection)
        record=dict(checkpoint=name,training_seconds=elapsed,environment_steps=steps,episodes=episodes,
                    evaluation_seconds=duration,stage=pool.stage,metrics=metrics,
                    selection_score=chosen_score,strong_balanced_score=strong_score,
                    checkpoint_selection=selection)
        if hasattr(learner,'diagnostics'): record['diagnostics']=learner.diagnostics()
        meta={**spec,**record}
        save(out/'checkpoints'/f'{name}.npz',weights,meta)
        if chosen_score > best_score:
            best_score,best_id=chosen_score,name
            save(out/'best.npz',weights,meta)
        if metrics['score'] > best_overall_score:
            best_overall_score,best_overall_id=metrics['score'],name
            save(out/'best_overall.npz',weights,meta)
        pool.add(f'{out.name}/{name}' if persist_training_state else name,weights,metrics['score'],factory)
        if persist_training_state:
            import torch
            bundle=dict(format_version=1,algorithm=learner.algorithm,board=asdict(board),
                        learner=learner.export_state(),pool=pool.export_state(),
                        open_rng_state=open_rng.getstate(),runner_rng_state=rng.getstate(),
                        cumulative_episodes=episode_offset+episodes,cumulative_steps=step_offset+steps)
            fd,tmp=tempfile.mkstemp(dir=out,suffix='.pt')
            os.close(fd)
            try:
                torch.save(bundle,tmp)
                os.replace(tmp,out/'resume_latest.pt')
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        append_jsonl(out/'evaluations.jsonl',record)
        write_json(out/f'evaluation_{eval_index:04d}.json',rows)
        eval_index+=1
        checkpoint_elapsed+=snapshot_time+time.perf_counter()-start
        if verbose:
            flood=metrics['by_opponent']['space']['win_rate']
            territory=metrics['by_opponent']['territory']['win_rate']
            print(f'{name}: train={elapsed:.1f}s steps={steps} overall={metrics["score"]:.1%} '
                  f'Flood-fill={flood:.1%} Territory={territory:.1%} '
                  f'{selection}={chosen_score:.1%}',flush=True)
    batch_size=getattr(learner,'batch_episodes',1)
    if cfg.max_episodes is not None and cfg.max_episodes<batch_size:
        raise ValueError('max_episodes must allow at least one complete batch')
    checkpoint() # baseline, zero training
    while elapsed<cfg.seconds and (cfg.max_episodes is None or episodes+batch_size<=cfg.max_episodes):
        target_index=stage_index(cfg,elapsed)
        if target_index!=current_index:
            current_index=target_index
            pool.stage=cfg.curriculum[current_index][0]
            stage_log.append(dict(stage=pool.stage,training_seconds=elapsed,environment_steps=steps,episodes=episodes))
            write_json(out/'stages.json',stage_log)
            if verbose: print(f'curriculum: {pool.stage} at {elapsed:.2f}s',flush=True)
        start=time.perf_counter()
        if hasattr(learner,'train_batch'):
            pool.progress=min(1.,elapsed/cfg.seconds)
            rows=learner.train_batch(board,pool,open_rng,progress=min(1.,elapsed/cfg.seconds))
        else:
            pool.progress=min(1.,elapsed/cfg.seconds)
            opening=training_opening(open_rng,episode_offset+episodes)
            opponent_name,opponent=pool.sample(board,opening.opponent_seed)
            result=play_episode(board,opening,learner,opponent,update=learner.update)
            rows=[dict(opponent=opponent_name,opening=asdict(opening),**asdict(result))]
        elapsed+=time.perf_counter()-start
        if getattr(learner, 'last_generation_record', None) is not None:
            append_jsonl(out/'generation_stats.jsonl',
                         {**learner.last_generation_record, 'training_seconds':elapsed,
                          'environment_steps_end':steps+sum(row['steps'] for row in rows)})
        for row in rows:
            episodes+=1; steps+=row['steps']
            append_jsonl(out/'episodes.jsonl',dict(episode=episodes,training_seconds=elapsed,
                          environment_steps=steps,stage=pool.stage,**row))
        stop=elapsed>=cfg.seconds or (cfg.max_episodes is not None and episodes+batch_size>cfg.max_episodes)
        if elapsed>=next_eval or stop or stage_index(cfg,elapsed)!=current_index:
            checkpoint()
            while next_eval<=elapsed: next_eval+=cfg.eval_interval
    summary=dict(algorithm=learner.algorithm,training_seconds=elapsed,requested_seconds=cfg.seconds,
                 evaluation_seconds=eval_elapsed,checkpoint_seconds=checkpoint_elapsed,
                 total_wall_seconds=time.perf_counter()-wall_start,episodes=episodes,environment_steps=steps,
                 evaluations=eval_index,best_checkpoint=best_id,best_score=best_score,
                 checkpoint_selection=selection,best_overall_checkpoint=best_overall_id,
                 best_overall_score=best_overall_score,
                 cumulative_episodes=episode_offset+episodes,cumulative_steps=step_offset+steps,
                 resume_from=str(resume_state_path) if resume_state_path else None,
                 stop_reason='time_budget' if elapsed>=cfg.seconds else 'episode_limit')
    write_json(out/'summary.json',summary)
    return summary
