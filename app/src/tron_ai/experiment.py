"""Sequential, equal-time experiments and comparative reports."""
from __future__ import annotations
import argparse, csv, json, math, os, platform, subprocess, sys
from pathlib import Path
from statistics import mean, stdev
from .train import parse_curriculum
from .training.runner import write_json
from .training.benchmark import HOLDOUT_SEED

ALGORITHMS=('dqn','ppo','evolution')
CURRICULUM='survival:0.4,space:0.25,competitive:0.15,selfplay:0.2'
PROFILES={
 'default':{a:[] for a in ALGORITHMS},
 'pilot':{'dqn':['--warmup','256','--epsilon-decay','10000'], 'ppo':['--entropy-coef','0.01','--rollout-steps','256'], 'evolution':['--population','16','--elites','4','--fitness-pairs','4']},
 'smoke':{'dqn':['--warmup','64','--batch-size','32','--epsilon-decay','500'], 'ppo':['--rollout-steps','64','--batch-size','32','--epochs','2'], 'evolution':['--population','4','--elites','1','--fitness-pairs','1']},
}

def read(path): return json.loads(path.read_text())
def history(path): return [json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()]

def command(args,algorithm,seed,path):
    cmd=[sys.executable,'-m','tron_ai.train',algorithm,'--seconds',str(args.seconds),'--eval-interval',str(args.eval_interval),'--eval-pairs',str(args.eval_pairs),'--seed',str(seed),'--stage','survival','--out',str(path)]
    if args.curriculum: cmd+=['--curriculum',args.curriculum]
    if algorithm!='evolution': cmd+=['--threads',str(args.threads)]
    return cmd+PROFILES[args.profile][algorithm]

def invoke(cmd,env,log):
    print('Running: '+' '.join(cmd),flush=True)
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in proc.stdout:
                f.write(line); f.flush(); print(line,end='',flush=True)
            if proc.wait(): raise RuntimeError(f'child failed; see {log}')
        finally:
            if proc.poll() is None: proc.terminate(); proc.wait()

def summarise(out):
    manifest=read(out/'manifest.json'); rows=[]; histories=[]
    for entry in manifest['runs']:
        path=out/entry['directory']; h=history(path); s=read(path/'summary.json')
        best=next(r for r in h if r['checkpoint']==s['best_checkpoint']); last=h[-1]
        row=dict(algorithm=entry['algorithm'],seed=entry['seed'],requested_seconds=s['requested_seconds'],training_seconds=s['training_seconds'],overshoot_seconds=max(0,s['training_seconds']-s['requested_seconds']),evaluation_seconds=s['evaluation_seconds'],checkpoint_seconds=s['checkpoint_seconds'],environment_steps=s['environment_steps'],episodes=s['episodes'],initial_score=h[0]['metrics']['score'],final_score=last['metrics']['score'],best_score=best['metrics']['score'],best_win_rate=best['metrics']['win_rate'],best_training_seconds=best['training_seconds'],best_checkpoint=best['checkpoint'])
        hold=path/'holdout'/'evaluation.json'
        if hold.exists():
            m=read(hold)['metrics']; row.update(holdout_score=m['score'],holdout_win_rate=m['win_rate'],holdout_draw_rate=m['draw_rate'],holdout_games=m['games'])
            row.update({f'holdout_{op}_win_rate':v['win_rate'] for op,v in m['by_opponent'].items()})
        rows.append(row); histories.append((entry,h))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    groups={}
    for alg in ALGORITHMS:
        subset=[r for r in rows if r['algorithm']==alg]
        if not subset: continue
        groups[alg]={'seeds':len(subset)}
        for k in ('best_score','final_score','holdout_score','holdout_win_rate','environment_steps','training_seconds'):
            v=[r[k] for r in subset if k in r]
            if v: groups[alg][k]={'mean':mean(v),'sample_sd':stdev(v) if len(v)>1 else None,'n':len(v)}
    write_json(out/'aggregate.json',groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=dict(zip(ALGORITHMS,['tab:blue','tab:green','tab:purple']))
    for entry,h in histories:
        a=entry['algorithm']; label=f"{a} seed {entry['seed']}"
        for ax,key,metric in [(axes[0,0],'training_seconds','score'),(axes[0,1],'environment_steps','score'),(axes[1,0],'training_seconds','win_rate')]:
            ax.plot([r[key] for r in h],[r['metrics'][metric] for r in h],marker='o',label=label,color=colors[a],alpha=.7)
    for ax,x,y in [(axes[0,0],'Active training seconds','Validation score'),(axes[0,1],'Environment steps','Validation score'),(axes[1,0],'Active training seconds','Validation win rate')]:
        ax.set(xlabel=x,ylabel=y,ylim=(0,1)); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    ax=axes[1,1]; available=[r for r in rows if 'holdout_win_rate' in r]
    if available:
        ax.bar(range(len(available)),[r['holdout_win_rate'] for r in available],color=[colors[r['algorithm']] for r in available]); ax.set_xticks(range(len(available)),[f"{r['algorithm']}\n{r['seed']}" for r in available],rotation=45,ha='right'); ax.set(ylabel='Selected checkpoint holdout win rate',ylim=(0,1))
    else: ax.text(.5,.5,'Holdout not run',ha='center',va='center',transform=ax.transAxes); ax.set_axis_off()
    fig.suptitle('Tron | serial training comparison (each seed shown)'); fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=300); p.add_argument('--eval-interval',type=float,default=60)
    p.add_argument('--eval-pairs',type=int,default=8); p.add_argument('--holdout-pairs',type=int,default=0,help='0 skips holdout; reserve for locked final protocol')
    p.add_argument('--holdout-seed',type=int,default=271828183,help='reserve this seed for the final locked experiment')
    p.add_argument('--seeds',type=int,nargs='+',default=[2026]); p.add_argument('--algorithms',choices=ALGORITHMS,nargs='+',default=list(ALGORITHMS))
    p.add_argument('--profile',choices=PROFILES,default='default'); p.add_argument('--curriculum',default=None,help=f'optional schedule, e.g. {CURRICULUM}')
    p.add_argument('--threads',type=int,default=1); p.add_argument('--out',type=Path,required=True); p.add_argument('--report-only',action='store_true')
    args=p.parse_args(argv); out=args.out.resolve()
    if args.report_only: summarise(out); return
    if not all(math.isfinite(v) and v>0 for v in (args.seconds,args.eval_interval)) or min(args.eval_pairs,args.threads)<1 or args.holdout_pairs<0 or len(set(args.seeds))!=len(args.seeds) or len(set(args.algorithms))!=len(args.algorithms) or min(args.seeds)<0: p.error('invalid budgets, thread count or duplicate/negative seeds')
    if args.curriculum:
        try: parse_curriculum(args.curriculum)
        except argparse.ArgumentTypeError as exc: p.error(str(exc))
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'output directory not empty: {out}')
    env=os.environ.copy(); env.update({k:str(args.threads) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}); env['MPLBACKEND']='Agg'
    runs=[]
    for i,seed in enumerate(args.seeds):
        order=args.algorithms[i%len(args.algorithms):]+args.algorithms[:i%len(args.algorithms)]
        for alg in order:
            directory=f'{alg}_seed{seed}'; runs.append(dict(algorithm=alg,seed=seed,directory=directory,command=command(args,alg,seed,out/directory)))
    write_json(out/'manifest.json',dict(runs=runs,profile=args.profile,curriculum=args.curriculum,eval_pairs=args.eval_pairs,holdout_pairs=args.holdout_pairs,holdout_seed=args.holdout_seed,threads=args.threads,platform=platform.platform(),processor=platform.processor(),python=sys.version,timing='serial subprocesses; training only; actual times reported'))
    for entry in runs: invoke(entry['command'],env,out/(entry['directory']+'.log'))
    if args.holdout_pairs:
        for entry in runs:
            path=out/entry['directory']; invoke([sys.executable,'-m','tron_ai.train','evaluate',str(path/'best.npz'),'--pairs',str(args.holdout_pairs),'--split','holdout','--holdout-seed',str(args.holdout_seed),'--threads',str(args.threads),'--out',str(path/'holdout')],env,out/(entry['directory']+'_holdout.log'))
    summarise(out)

if __name__=='__main__': main()
