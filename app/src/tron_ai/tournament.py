"""Paired round-robin benchmark for the original scripted policies."""
from __future__ import annotations
import argparse, csv, itertools, json, platform, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tron_ai.engine import TronConfig, TronEnv
from tron_ai.opponents import OPPONENTS, ScriptedPolicy

@dataclass(frozen=True)
class Match:
    a: str
    b: str
    opening: int
    axis: str
    board_seed: int
    a_seed: int
    b_seed: int
    swap: bool
    width: int = 30
    height: int = 24

def schedule(pairs, seed, width=30, height=24):
    """Reuse every opening for each matchup and both side assignments."""
    if pairs < 1: raise ValueError('pairs must be positive')
    TronConfig(width,height)
    rng=random.Random(seed)
    openings=[(i,('horizontal','vertical')[i%2],rng.getrandbits(63),
               {kind:rng.getrandbits(63) for kind in OPPONENTS}) for i in range(pairs)]
    for a,b in itertools.combinations(OPPONENTS,2):
        for i,axis,board_seed,seeds in openings:
            for swap in (False,True):
                yield Match(a,b,i,axis,board_seed,seeds[a],seeds[b],swap,width,height)

def play_match(m):
    config=TronConfig(m.width,m.height); env=TronEnv(config)
    env.reset(seed=m.board_seed,mode='random_mirrored',axis=m.axis,swap_sides=m.swap)
    initial=env.state.heads
    policies=[ScriptedPolicy(m.a,seed=m.a_seed),ScriptedPolicy(m.b,seed=m.b_seed)]
    cpu_ns=[0,0]
    while not env.state.done:
        state=env.state; actions=[]
        for i,policy in enumerate(policies):
            start=time.process_time_ns()
            actions.append(policy.choose_action(state,i,config))
            cpu_ns[i]+=time.process_time_ns()-start
        env.step(actions)
    return {**asdict(m),'a_start':str(initial[0]),'b_start':str(initial[1]),
            'outcome':'draw' if env.state.winner is None else ('a' if env.state.winner==0 else 'b'),
            'steps':env.state.step_count,'end_reason':env.state.end_reason.value,
            'a_cpu_ns':cpu_ns[0],'b_cpu_ns':cpu_ns[1]}

def summarise(rows):
    head=[]
    ranking={p:dict(agent=p,wins=0,losses=0,draws=0,games=0,steps=0,cpu_ns=0) for p in OPPONENTS}
    for a,b in itertools.combinations(OPPONENTS,2):
        games=[r for r in rows if r['a']==a and r['b']==b]
        n=len(games)
        if not n: continue
        wins=sum(r['outcome']=='a' for r in games); losses=sum(r['outcome']=='b' for r in games)
        draws=n-wins-losses; steps=sum(r['steps'] for r in games)
        head.append(dict(a=a,b=b,games=n,a_wins=wins,b_wins=losses,draws=draws,
                         a_win_rate=wins/n,b_win_rate=losses/n,draw_rate=draws/n,average_steps=steps/n))
        for p,w,l,side in ((a,wins,losses,'a'),(b,losses,wins,'b')):
            s=ranking[p]; s['wins']+=w; s['losses']+=l; s['draws']+=draws; s['games']+=n; s['steps']+=steps
            s['cpu_ns']+=sum(r[f'{side}_cpu_ns'] for r in games)
    result=[]
    for s in ranking.values():
        n=s['games']
        if n: result.append({**s,'win_rate':s['wins']/n,'draw_rate':s['draws']/n,'loss_rate':s['losses']/n,
                             'score':(s['wins']+.5*s['draws'])/n,'mean_reward':(s['wins']-s['losses'])/n,
                             'average_steps':s['steps']/n,'cpu_ms_per_move':s['cpu_ns']/s['steps']/1e6})
    return head,sorted(result,key=lambda s:(-s['score'],s['agent']))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def plot(out,head,ranking):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    kinds=list(OPPONENTS); labels=['Random','Random safe','Flood-fill','Territory']
    matrix=np.full((4,4),np.nan)
    for r in head:
        a,b=kinds.index(r['a']),kinds.index(r['b'])
        matrix[a,b]=100*r['a_win_rate']; matrix[b,a]=100*r['b_win_rate']
    fig,ax=plt.subplots(figsize=(7.5,5.6),layout='constrained')
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis'); im.cmap.set_bad('#626b75')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,
           xlabel='Opponent',ylabel='Agent',title='Tron | Head-to-head win rate (%)')
    for i in range(4):
        for j in range(4):
            v=matrix[i,j]; ax.text(j,i,'—' if np.isnan(v) else f'{v:.1f}%',ha='center',va='center',color='white' if np.isnan(v) or v<50 else 'black')
    fig.colorbar(im,ax=ax,label='Outright wins (%)'); fig.savefig(out/'win_rate_heatmap.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    names=[OPPONENTS[r['agent']] for r in ranking]; left=np.zeros(len(ranking))
    for field,label in [('win_rate','Win'),('draw_rate','Draw'),('loss_rate','Loss')]:
        vals=np.array([100*r[field] for r in ranking]); ax.barh(names,vals,left=left,label=label); left+=vals
    ax.set(xlim=(0,100),xlabel='Matches (%)',title='Results against the other three scripted agents'); ax.invert_yaxis(); ax.legend(loc='lower right')
    fig.savefig(out/'outcomes.png',dpi=180); plt.close(fig)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',type=int,default=100,help='Mirrored opening pairs per distinct matchup (two games each)')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--workers',type=int,default=1,help='1 for isolated timing; parallel runs measure process CPU time')
    parser.add_argument('--width',type=int,default=30); parser.add_argument('--height',type=int,default=24)
    parser.add_argument('--out',type=Path,default=Path('results/tournament')); parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    if args.pairs<1 or args.workers<1: parser.error('pairs and workers must be positive')
    try: matches=list(schedule(args.pairs,args.seed,args.width,args.height))
    except ValueError as exc: parser.error(str(exc))
    args.out.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter(); rows=[]; print(f'{len(matches)} games; {args.workers} worker(s)',flush=True)
    with (args.out/'matches.csv').open('w',newline='') as f:
        def collect(results):
            writer=None
            for i,r in enumerate(results,1):
                if writer is None: writer=csv.DictWriter(f,fieldnames=list(r)); writer.writeheader()
                writer.writerow(r); f.flush(); rows.append(r)
                if i%100==0 or i==len(matches): print(f'{i}/{len(matches)} games',flush=True)
        if args.workers==1: collect(map(play_match,matches))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool: collect(pool.map(play_match,matches,chunksize=1))
    head,ranking=summarise(rows); write_csv(args.out/'matchups.csv',head); write_csv(args.out/'ranking.csv',ranking)
    metadata={'seed':args.seed,'pairs_per_matchup':args.pairs,'workers':args.workers,'width':args.width,'height':args.height,
              'games':len(rows),'wall_seconds':time.perf_counter()-t,'python':sys.version,'platform':platform.platform(),
              'timing':'process CPU time around choose_action only; not wall latency; excludes engine and rendering'}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if not args.no_plots: plot(args.out,head,ranking)
    for r in ranking: print(f"{r['agent']:10s} W/D/L {r['wins']}/{r['draws']}/{r['losses']} score {r['score']:.1%} CPU {r['cpu_ms_per_move']:.3f} ms/move")
    print(f'Saved to {args.out}')

if __name__=='__main__': main()
