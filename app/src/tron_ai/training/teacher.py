"""Demonstration collection and action-label pretraining; never uses privileged inputs."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json,random,time
import numpy as np
import torch
from torch.nn import functional as F
from tron_ai.engine import TronConfig,TronEnv
from tron_ai.opponents import ScriptedPolicy
from tron_ai.observations import observation_shape
from .checkpoints import save
from .advanced_dqn import DuelingNetwork,ARCHITECTURE as DQN_ARCH
from .ppo import ActorCritic,ARCHITECTURE as PPO_ARCH

DATA_VERSION=1


def collect(board:TronConfig, count:int, seed:int, out:Path):
    if count<2:raise ValueError('need at least 2 samples')
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    rng=random.Random(seed)
    shape=observation_shape(board)
    obs=[];actions=[];episode_ids=[];teacher_kinds=[]
    episode=0
    started=time.perf_counter()
    while len(actions)<count:
        env=TronEnv(board)
        env.reset(seed=rng.getrandbits(63),mode='random_mirrored',
                  axis=rng.choice(('horizontal','vertical')),swap_sides=False)
        player=episode%2
        kind=rng.choices(['space','territory'],weights=[.65,.35],k=1)[0]
        opponent_kind=rng.choices(['random','safe','space'],weights=[.3,.5,.2],k=1)[0]
        teacher=ScriptedPolicy(kind,seed=rng.getrandbits(63))
        opponent=ScriptedPolicy(opponent_kind,seed=rng.getrandbits(63))
        while not env.state.done and len(actions)<count:
            observation=env.observe(player)
            action=teacher.choose_action(env.state,player,board)
            other=opponent.choose_action(env.state,1-player,board)
            obs.append(np.packbits(observation.reshape(-1)))
            actions.append(int(action));episode_ids.append(episode)
            teacher_kinds.append(0 if kind=='space' else 1)
            env.step((action,other) if player==0 else (other,action))
        episode+=1
    payload=json.dumps(dict(format_version=DATA_VERSION,board=asdict(board),shape=list(shape),
                            action_labels={'left':0,'straight':1,'right':2},seed=seed,
                            teachers=['space','territory'],examples=count,episodes=episode))
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,obs=np.asarray(obs,dtype=np.uint8),actions=np.asarray(actions,dtype=np.int64),
                        episode_ids=np.asarray(episode_ids,dtype=np.int32),teacher_kinds=np.asarray(teacher_kinds,dtype=np.uint8),
                        metadata=np.array(payload))
    return dict(dataset=str(out),examples=len(actions),episodes=episode,
                teacher_counts={'space':teacher_kinds.count(0),'territory':teacher_kinds.count(1)},
                wall_seconds=time.perf_counter()-started)


def load_dataset(path):
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(f['metadata'].item());obs=f['obs'].copy();actions=f['actions'].copy();episodes=f['episode_ids'].copy()
    if meta['format_version']!=DATA_VERSION:raise ValueError('unsupported teacher dataset')
    shape=tuple(meta['shape'])
    if obs.shape!=(len(actions),(int(np.prod(shape))+7)//8) or episodes.shape!=actions.shape:
        raise ValueError('teacher dataset shape mismatch')
    if not np.isin(actions,np.arange(3)).all():raise ValueError('teacher action outside 0..2')
    return obs,actions,episodes,meta


def pretrain(dataset:Path,algo:str,steps:int,seed:int,out:Path,batch:int=64,lr:float=.0001,threads:int=1,initial_weights=None):
    if algo not in ('dqn','ppo'):raise ValueError('only dqn and ppo support demonstrations')
    if steps<1 or batch<1 or threads<1:raise ValueError('steps, batch, threads must be positive')
    torch.set_num_threads(threads)
    obs,actions,episodes,meta=load_dataset(dataset)
    board=TronConfig(**meta['board']);shape=tuple(meta['shape'])
    rng=np.random.default_rng(seed)
    unique=np.unique(episodes);rng.shuffle(unique)
    split=max(1,min(len(unique)-1,int(.9*len(unique))))
    if len(unique)<2:raise ValueError('at least two demonstration episodes are required')
    train_idx=np.flatnonzero(np.isin(episodes,unique[:split]));val_idx=np.flatnonzero(np.isin(episodes,unique[split:]))
    if not len(train_idx) or not len(val_idx):raise ValueError('empty demonstration split')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network=DuelingNetwork() if algo=='dqn' else ActorCritic()
    if initial_weights is not None:
        if algo=='dqn':
            from .advanced_dqn import upgrade_legacy_weights,load_network
            network.load_state_dict(load_network(upgrade_legacy_weights(initial_weights)).state_dict())
        else:
            from .ppo import load_network
            network.load_state_dict(load_network(initial_weights).state_dict())
    optimizer=torch.optim.Adam(network.parameters(),lr=lr)
    def unpack(indices):
        return torch.from_numpy(np.unpackbits(obs[indices],axis=1,count=int(np.prod(shape))).reshape(len(indices),*shape)).float()
    def classify(x):
        values=network(x)
        return values[0] if isinstance(values,tuple) else values
    best=-1.;best_step=0;last_loss=None
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for step in range(1,steps+1):
        ids=rng.choice(train_idx,size=batch,replace=True)
        network.train()
        logits=classify(unpack(ids));labels=torch.from_numpy(actions[ids])
        loss=F.cross_entropy(logits,labels)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();last_loss=float(loss.detach())
        if step==1 or step==steps or step%max(1,steps//10)==0:
            network.eval()
            accuracy=[]
            with torch.no_grad():
                for subset in np.array_split(val_idx,max(1,int(np.ceil(len(val_idx)/128)))):
                    if len(subset):accuracy.append((classify(unpack(subset)).argmax(1)==torch.from_numpy(actions[subset])).float().mean().item()*len(subset))
            score=sum(accuracy)/len(val_idx)
            if score>best:
                best=score;best_step=step
                weights={k:v.detach().numpy().copy() for k,v in network.state_dict().items()}
                info=dict(algorithm=algo,board=asdict(board),observation_shape=list(shape),
                          source='teacher_imitation',architecture=DQN_ARCH if algo=='dqn' else PPO_ARCH,
                          dataset=str(dataset),warm_start=initial_weights is not None,validation_episodes=len(unique)-split,
                          validation_action_accuracy=best,step=step)
                save(out/'best.npz',weights,info)
            print(f'{algo} imitation step={step}/{steps} held-out teacher accuracy={score:.1%}',flush=True)
    result=dict(algorithm=algo,steps=steps,best_step=best_step,heldout_teacher_accuracy=best,
                train_samples=len(train_idx),holdout_samples=len(val_idx),last_training_loss=last_loss,
                checkpoint=str(out/'best.npz'))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
