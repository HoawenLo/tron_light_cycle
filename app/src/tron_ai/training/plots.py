"""Optional offline diagnostics: common primary and secondary comparison axes."""
from pathlib import Path
import json

def plot(run: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    fig.suptitle('Tron | fixed reference evaluation — '+json.loads((run/'config.json').read_text())['algorithm'])
    t=[r['training_seconds'] for r in records]; steps=[r['environment_steps'] for r in records]
    for ax,x,label in ((axes[0,0],t,'Active training seconds'),(axes[0,1],steps,'Environment steps')):
        for metric in ('win_rate','draw_rate','loss_rate','score'):
            ax.plot(x,[r['metrics'][metric] for r in records],marker='o',label=metric.replace('_',' '))
        ax.set(xlabel=label,ylabel='Fraction',ylim=(0,1)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    for op in records[0]['metrics']['by_opponent']:
        axes[1,0].plot(t,[r['metrics']['by_opponent'][op]['win_rate'] for r in records],marker='o',label=op)
    axes[1,0].set(xlabel='Active training seconds',ylabel='Win rate by opponent',ylim=(0,1)); axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=.2)
    for i,a in enumerate(('left','straight','right')):
        axes[1,1].plot(t,[r['metrics']['action_fractions'][i] for r in records],marker='o',label=a)
    axes[1,1].set(xlabel='Active training seconds',ylabel='Action fraction',ylim=(0,1)); axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=.2)
    out=run/'learning_curves.png'; fig.savefig(out,dpi=150); plt.close(fig)
    if any('policy_entropy' in r.get('diagnostics',{}) for r in records):
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        fig.suptitle('PPO diagnostics | per evaluation interval')
        def values(key): return [r.get('diagnostics',{}).get(key) for r in records]
        axes[0,0].plot(t,values('policy_entropy'),marker='o',label='collection entropy')
        import math
        axes[0,0].axhline(math.log(3),ls='--',label='uniform maximum')
        axes[0,0].set(ylabel='Entropy (nats)',ylim=(0,1.2)); axes[0,0].legend()
        for i,a in enumerate(('left','straight','right')):
            axes[0,1].plot(t,[v[i] if v is not None else None for v in values('training_action_fractions')],marker='o',label=a)
        axes[0,1].set(ylabel='Training action fraction',ylim=(0,1)); axes[0,1].legend()
        for key in ('policy_loss','value_loss'):
            axes[1,0].plot(t,values(key),marker='o',label=key)
        axes[1,0].legend(); axes[1,0].set(ylabel='Loss')
        axes[1,1].plot(t,values('approx_kl'),marker='o',label='approx KL')
        axes[1,1].plot(t,values('clip_fraction'),marker='o',label='clip fraction'); axes[1,1].legend()
        for ax in axes.flat: ax.set_xlabel('Active training seconds'); ax.grid(alpha=.2)
        fig.savefig(run/'ppo_diagnostics.png',dpi=150); plt.close(fig)
    if any('generation' in r.get('diagnostics',{}) for r in records):
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        d=[r.get('diagnostics',{}) for r in records]
        for key in ('best_fitness','mean_fitness','worst_fitness'):
            axes[0].plot(t,[v.get(key) for v in d],marker='o',label=key)
        axes[0].set(xlabel='Active training seconds',ylabel='Generation mean reward',ylim=(-1,1)); axes[0].legend(fontsize=8)
        axes[1].plot(t,[v.get('generation') for v in d],marker='o'); axes[1].set(xlabel='Active training seconds',ylabel='Completed generations')
        for ax in axes: ax.grid(alpha=.2)
        fig.savefig(run/'evolution_diagnostics.png',dpi=150); plt.close(fig)
    generations = run/'generation_stats.jsonl'
    if generations.exists():
        rows=[json.loads(line) for line in generations.read_text().splitlines()]
        if rows:
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            x=[r['generation'] for r in rows]
            for key in ('best_fitness','mean_fitness','worst_fitness'):
                axes[0,0].plot(x,[r[key] for r in rows],label=key,alpha=.85)
            axes[0,0].set(ylabel='Fitness (mean terminal reward)',ylim=(-1,1))
            axes[0,0].legend(fontsize=8)
            axes[0,1].plot(x,[r['fitness_std'] for r in rows])
            axes[0,1].set(ylabel='Within-generation fitness std')
            names=sorted(set(name for r in rows for name in r['opponent_pairs']))
            for name in names:
                axes[1,0].plot(x,[r['opponent_pairs'].get(name,0) for r in rows],label=name)
            axes[1,0].set(ylabel='Paired openings by opponent')
            axes[1,0].legend(fontsize=8)
            for name in names:
                axes[1,1].plot(x,[r['opponent_mean_reward'].get(name,float('nan')) for r in rows],label=name)
            axes[1,1].set(ylabel='Population mean reward by opponent',ylim=(-1,1))
            axes[1,1].legend(fontsize=8)
            for ax in axes.flat:
                ax.set_xlabel('Generation'); ax.grid(alpha=.2)
            fig.suptitle('Neuroevolution | population and balanced opponent diagnostics')
            fig.savefig(run/'evolution_generations.png',dpi=150)
            plt.close(fig)
            # Phase 5.4: ES/ARS-specific search diagnostics, not mutation-schedule plots.
            if all(r.get('optimizer') in ('es','ars') for r in rows):
                fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
                for metric in ('best_fitness','mean_fitness','worst_fitness'):
                    axes[0,0].plot(x,[r[metric] for r in rows],label=metric)
                axes[0,0].set(ylabel='Perturbed candidate reward',ylim=(-1,1))
                axes[0,0].legend(fontsize=8)
                axes[0,1].plot(x,[r['update_l2'] for r in rows],label='Actual update L2')
                axes[0,1].plot(x,[r['unclipped_update_l2'] for r in rows],alpha=.6,label='Before clipping')
                axes[0,1].set(ylabel='Parameter update L2 norm')
                axes[0,1].legend(fontsize=8)
                axes[1,0].plot(x,[r['signal_fraction'] for r in rows])
                axes[1,0].set(ylabel='Directions with unequal +/- scores',ylim=(0,1))
                for name in names:
                    axes[1,1].plot(x,[r['opponent_mean_reward'].get(name,float('nan')) for r in rows],label=name)
                axes[1,1].set(ylabel='Mean reward by opponent',ylim=(-1,1))
                axes[1,1].legend(fontsize=8)
                for ax in axes.flat:
                    ax.set(xlabel='Search iteration'); ax.grid(alpha=.2)
                fig.suptitle('Antithetic ES / ARS | CPU optimisation diagnostics')
                fig.savefig(run/'search_diagnostics.png',dpi=150)
                plt.close(fig)
            # Phase 5.2: mutation schedule and best-candidate per-opponent
            # outcomes complement (but do not replace) fixed validation plots.
            if all(r.get('next_offspring_mutation_sigma') is not None for r in rows):
                fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
                axes[0].plot(x,[r['next_offspring_mutation_sigma'] for r in rows])
                axes[0].set(ylabel='Sigma used to create next generation',xlabel='Generation')
                for name in names:
                    axes[1].plot(x,[r.get('best_candidate_by_opponent',{}).get(name,{}).get('mean_reward',float('nan')) for r in rows],label=name)
                axes[1].set(ylabel='Best candidate fitness by opponent',xlabel='Generation',ylim=(-1,1))
                axes[1].legend(fontsize=8)
                for name in names:
                    axes[2].plot(x,[r['opponent_pairs'].get(name,0) for r in rows],label=name)
                axes[2].set(ylabel='Opponent pairs / generation',xlabel='Generation')
                axes[2].legend(fontsize=8)
                for ax in axes: ax.grid(alpha=.2)
                fig.suptitle('Phase 5.2 | mutation and opponent breakdown')
                fig.savefig(run/'evolution_targeted.png',dpi=150)
                plt.close(fig)
    return out
