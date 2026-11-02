"""Dense reward-to-go, leave-one-out REINFORCE; no critic, PPO or data reuse."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from common import ROOT, atomic_json, atomic_torch, autocast, cpu, digest, file_hash, source_id, frozen_teacher
from environment import COST_VERSION, exact_counts, reorder, swap_permutations
from model import SwapPolicy

def reward_and_advantage(start_cost, costs, best_costs, group):
    """All tensors detached; costs are [T,M], M graph-major then rollout-major."""
    assert group>=2 and costs.ndim==2 and costs.shape==best_costs.shape
    assert costs.shape[1]%group==0
    before=torch.cat([start_cost[None],costs[:-1]],0).double()
    rewards=(before-costs.double())/start_cost.double()[None]
    rewards[-1]+=(costs[-1].double()-best_costs[-1].double())/start_cost.double()
    returns=rewards.flip(0).cumsum(0).flip(0)
    values=returns.reshape(len(costs),-1,group)
    baseline=(values.sum(-1,keepdim=True)-values)/(group-1)
    advantage=(values-baseline).reshape_as(returns)
    expected=(start_cost.double()-best_costs[-1].double())/start_cost.double()
    assert torch.allclose(rewards.sum(0),expected,atol=1e-10,rtol=1e-10)
    assert torch.allclose(returns,(before-best_costs[-1].double()[None])/start_cost.double()[None],atol=1e-10,rtol=1e-10)
    return rewards.float(),returns.float(),advantage.float().detach()

@torch.no_grad()
def collect(policy,graphs,starts,horizon,group,generator,config,device,capture=True):
    base=graphs.repeat_interleave(group,0)
    order=starts.long().repeat_interleave(group,0).clone()
    initial=exact_counts(reorder(base,order))[0]
    best=initial.clone();best_order=order.clone()
    orders=[];actions=[];logps=[];fractions=[];costs=[];bests=[]
    zero=torch.zeros(len(order),device=device)
    for t in range(horizon):
        fraction=best.float()/initial
        with autocast(device,config):logits=policy(reorder(base,order),zero,fraction)
        noise=torch.rand(logits.shape,generator=generator,device=device).clamp_(1e-7,1-1e-7)
        action=(logits-noise.log().neg().log()).argmax(-1)
        if capture:
            orders.append(order.to(torch.int16));fractions.append(fraction)
            logps.append(logits.log_softmax(-1).gather(1,action[:,None]).squeeze(1))
        actions.append(action.to(torch.int16))
        order=swap_permutations(order,policy.pairs[action])
        current=exact_counts(reorder(base,order))[0]
        improved=current<best
        best_order=torch.where(improved[:,None],order,best_order)
        best=torch.minimum(best,current)
        costs.append(current.to(torch.int32));bests.append(best.to(torch.int32))
    result=dict(graphs=graphs,starts=starts.to(torch.int16),initial=initial,actions=torch.stack(actions),
                costs=torch.stack(costs),best_costs=torch.stack(bests),best_order=best_order.to(torch.int16),
                final_order=order.to(torch.int16),group=group,horizon=horizon)
    if capture:result.update(orders=torch.stack(orders),fractions=torch.stack(fractions),old_logps=torch.stack(logps))
    assert torch.equal(exact_counts(reorder(base,best_order))[0],best)
    return result

def update(network,policy,optimizer,trace,config,device,distributed=False):
    rewards,returns,advantages=reward_and_advantage(trace['initial'],trace['costs'],trace['best_costs'],trace['group'])
    horizon,count,n=trace['orders'].shape;total=horizon*count
    actions=trace['actions'].reshape(-1).long();orders=trace['orders'].reshape(-1,n)
    fractions=trace['fractions'].reshape(-1);old=trace['old_logps'].reshape(-1)
    advantages=advantages.reshape(-1)
    optimizer.zero_grad(set_to_none=True)
    loss_sum=torch.zeros((),device=device);entropy_sum=loss_sum.clone();error=loss_sum.clone()
    for lo in range(0,total,config['microbatch']):
        hi=min(lo+config['microbatch'],total)
        context=network.no_sync() if distributed and hi<total else nullcontext()
        with context:
            flat_ids=torch.arange(lo,hi,device=device)
            graph_ids=(flat_ids%count)//trace['group']
            matrix=reorder(trace['graphs'][graph_ids],orders[lo:hi].long())
            with autocast(device,config):logits=network(matrix,torch.zeros(hi-lo,device=device),fractions[lo:hi])
            lp=logits.log_softmax(-1)
            selected=lp.gather(1,actions[lo:hi,None]).squeeze(1)
            loss=-(selected*advantages[lo:hi]).sum()/total
            loss.backward()
        loss_sum+=loss.detach()
        entropy_sum+=(-(lp.detach().exp()*lp.detach()).sum(-1)).sum()/total
        error+=(selected.detach()-old[lo:hi]).abs().sum()/total
    norm=nn.utils.clip_grad_norm_(policy.parameters(),config['gradient_clip'])
    finite=torch.tensor(int(torch.isfinite(loss_sum) and torch.isfinite(norm)),device=device)
    if distributed:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
    if not bool(finite):raise FloatingPointError('Nonfinite gradient/loss; no optimizer step performed')
    # Different bf16 GEMM batch sizes can produce small rounding differences.
    if float(error)>0.05:raise RuntimeError('Collection/recomputed logp mismatch >0.05; inspect replay')
    optimizer.step()
    before=torch.cat([trace['initial'][None],trace['costs'][:-1]],0)
    immediate=before-trace['costs']
    return dict(loss=float(loss_sum),entropy=float(entropy_sum),grad_norm=float(norm),
                recompute_logp_abs_delta=float(error),advantage_std=float(advantages.std()),
                zero_advantage_fraction=float((advantages.abs()<1e-10).float().mean()),
                positive_delta_fraction=float((immediate>0).float().mean()),
                negative_delta_fraction=float((immediate<0).float().mean()),
                zero_delta_fraction=float((immediate==0).float().mean()),
                mean_reward=float(returns[0].mean()),
                mean_initial_cost=float(trace['initial'].float().mean()),
                mean_best_cost=float(trace['best_costs'][-1].float().mean()),
                mean_group_best_cost=float(trace['best_costs'][-1].reshape(-1,trace['group']).min(-1).values.float().mean()))

def barrier(world):
    if world>1:dist.barrier()

def max_time(seconds,device,world):
    value=torch.tensor(seconds,device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return float(value)

def reduced_metrics(metrics,device,world):
    keys=sorted(metrics);values=torch.tensor([metrics[k] for k in keys],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(values);values/=world
    return dict(zip(keys,values.cpu().tolist()))

def initial_model(config,device):
    assert file_hash(config['initial_checkpoint'])==config['initial_checkpoint_sha256'],'Initial checkpoint changed'
    checkpoint=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=False)
    assert checkpoint['step']==config['initial_step']
    policy=SwapPolicy(config['side'],config['width'],config['layers'],config['head_width']).to(device)
    policy.load_state_dict(checkpoint['model'],strict=True)
    return policy

def initialize(config,device):
    return initial_model(config,device)

@torch.no_grad()
def evaluate(policy,split,config,device,rank,world,run,label,step,test=False):
    folder=run/'evaluation'/label;folder.mkdir(parents=True,exist_ok=True)
    own=list(range(rank,len(split['graphs']),world));rows=[]
    group=config['test_group'] if test else config['evaluation_group']
    chunk=config['test_graph_batch'] if test else config['evaluation_graph_batch']
    budgets=config['test_budgets'] if test else config['validation_budgets']
    began=time.perf_counter();policy.eval()
    seed=config['test_seed']+500000 if test else config['validation_seed']+500000
    for lo in range(0,len(own),chunk):
        ids=own[lo:lo+chunk]
        rng=torch.Generator(device=device).manual_seed(seed+rank*10000+lo)
        graph=split['graphs'][ids].to(device);start=split['starts'][ids].to(device)
        trace=collect(policy,graph,start,max(budgets),group,rng,config,device,capture=False)
        for j,index in enumerate(ids):
            initial=int(trace['initial'][j*group])
            for budget in budgets:
                costs=trace['best_costs'][budget-1].reshape(-1,group)[j]
                rows.append(dict(graph=index,budget=budget,initial=initial,cost=int(costs.min()),
                                 first4_cost=int(costs[:4].min()),single_cost=int(costs[0]),group=group))
        if test:
            atomic_torch(folder/f'step{step:06d}_rank{rank}_block{lo//chunk:04d}.pt',
                         cpu(dict(ids=ids,starts=start,initial=trace['initial'],actions=trace['actions'],
                                  costs=trace['costs'],best_costs=trace['best_costs'],seed=seed+rank*10000+lo)))
        del trace
    atomic_json(folder/f'per_graph_step{step:06d}_rank{rank}.json',rows)
    barrier(world);result=None
    if rank==0:
        all_rows=[]
        for r in range(world):all_rows+=json.loads((folder/f'per_graph_step{step:06d}_rank{r}.json').read_text())
        assert len(all_rows)==len(split['graphs'])*len(budgets)
        by_budget={}
        for budget in budgets:
            values=sorted([v for v in all_rows if v['budget']==budget],key=lambda x:x['graph'])
            assert [v['graph'] for v in values]==list(range(len(split['graphs'])))
            mean=lambda key:sum(v[key] for v in values)/len(values)
            by_budget[str(budget)]=dict(cost=mean('cost'),first4_cost=mean('first4_cost'),single_cost=mean('single_cost'),
                                        initial=mean('initial'),normalized=mean('cost')/mean('initial'))
        result=dict(step=step,graphs=len(split['graphs']),group=group,by_budget=by_budget,seconds=time.perf_counter()-began)
        atomic_json(folder/f'summary_step{step:06d}.json',result)
    barrier(world)
    if world>1:
        objects=[result];dist.broadcast_object_list(objects,src=0);result=objects[0]
    policy.train();return result

def save_checkpoint(policy,optimizer,step,best,elapsed,rng,graph_rng,identity,config,run,rank,world,device):
    local=dict(sampler=rng.get_state().cpu(),graph_rng=graph_rng.get_state(),torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None)
    states=[None]*world
    if world>1:dist.all_gather_object(states,local)
    else:states=[local]
    # Verify all replicas have the same parameters before committing.
    checksum=torch.stack([p.detach().double().sum() for p in policy.parameters()]).sum()
    if world>1:
        values=[torch.empty_like(checksum) for _ in range(world)];dist.all_gather(values,checksum)
        assert all(torch.allclose(v,values[0],atol=1e-9,rtol=1e-12) for v in values),'DDP replica divergence'
    if rank==0:
        path=run/'checkpoints'/f'step{step:06d}.pt'
        best_snapshot=torch.load(run/'best_model.pt',map_location='cpu',weights_only=False) if (run/'best_model.pt').exists() else None
        atomic_torch(path,dict(model=cpu(policy.state_dict()),optimizer=cpu(optimizer.state_dict()),step=step,
                              best=best,elapsed_training_seconds=elapsed,rank_states=states,
                              identity=identity,config=config,world=world,best_model=best_snapshot))
        atomic_json(run/'latest_checkpoint.json',dict(step=step,file=str(path),sha256=file_hash(path)))
        files=sorted((run/'checkpoints').glob('step*.pt'))
        for old in files[:-2]:old.unlink()
    barrier(world)

def restore(policy,optimizer,rng,graph_rng,identity,run,rank,world,device):
    pointer=run/'latest_checkpoint.json'
    if not pointer.exists():return 0,dict(cost=float('inf'),step=0),0.
    saved=json.loads(pointer.read_text());assert file_hash(saved['file'])==saved['sha256']
    state=torch.load(saved['file'],map_location='cpu',weights_only=False)
    assert state['identity']==identity and state['world']==world
    policy.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    if rank==0 and state.get('best_model') is not None:
        assert state['best_model']['step']==state['best']['step']
        atomic_torch(run/'best_model.pt',state['best_model'])
    barrier(world)
    local=state['rank_states'][rank]
    rng.set_state(local['sampler']);graph_rng.set_state(local['graph_rng']);torch.set_rng_state(local['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(local['cuda'],device)
    return state['step'],state['best'],state['elapsed_training_seconds']

def paired_comparison(run,student_label,student_step,reference_label,reference_step,world,budgets):
    import statistics
    def load(label,step):
        rows=[]
        for rank in range(world):rows+=json.loads((run/'evaluation'/label/f'per_graph_step{step:06d}_rank{rank}.json').read_text())
        return {(r['graph'],r['budget']):r for r in rows}
    student=load(student_label,student_step);reference=load(reference_label,reference_step)
    assert student.keys()==reference.keys();result={}
    for budget in budgets:
        keys=sorted(k for k in student if k[1]==budget)
        differences=[(reference[k]['cost']-student[k]['cost'])/reference[k]['cost'] for k in keys]
        se=statistics.stdev(differences)/math.sqrt(len(keys)) if len(keys)>1 else 0.
        mean=statistics.mean(differences)
        result[str(budget)]=dict(relative_improvement=mean,ci95=[mean-1.96*se,mean+1.96*se],
                                wins=sum(student[k]['cost']<reference[k]['cost'] for k in keys),graphs=len(keys))
    return result

def final_heuristics(split,config,run):
    from collections import deque
    from teacher import generate
    rows=[]
    for index in range(len(split['graphs'])):
        graph=split['graphs'][index];start=split['starts'][index].long()
        matrix=reorder(graph[None],start[None])[0];side=config['side']
        neighbours=[matrix[u].nonzero().flatten().tolist() for u in range(2*side)]
        values={}
        for name,dfs in [('BFS',False),('DFS',True)]:
            seen=set();order=[]
            for root in range(2*side):
                if root in seen:continue
                if dfs:
                    stack=[root]
                    while stack:
                        u=stack.pop()
                        if u in seen:continue
                        seen.add(u);order.append(u)
                        stack.extend(v for v in reversed(neighbours[u]) if v not in seen)
                else:
                    seen.add(root);queue=deque([root])
                    while queue:
                        u=queue.popleft();order.append(u)
                        for v in neighbours[u]:
                            if v not in seen:seen.add(v);queue.append(v)
            perm=torch.tensor([u for u in order if u<side]+[u for u in order if u>=side])
            values[name]=int(exact_counts(reorder(matrix[None],perm[None]))[0][0])
        teacher=generate(int(split['seeds'][index]),side)
        assert torch.equal(teacher['graph'],graph) and torch.equal(teacher['start'].long(),start)
        values['teacher']=int(teacher['costs'][-1]);values['initial']=int(split['initial'][index])
        rows.append(dict(graph=index,**values))
    result=dict(graphs=len(rows),means={k:sum(r[k] for r in rows)/len(rows) for k in ['initial','teacher','BFS','DFS']},records=rows)
    atomic_json(run/'evaluation/test_heuristics.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'rl_config.json'))
    p.add_argument('--data',default=str(ROOT/'data'));p.add_argument('--run',required=True)
    p.add_argument('--overrides',default='{}');p.add_argument('--max-new-updates',type=int,default=0)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());config.update(json.loads(args.overrides))
    if config['initial_checkpoint_sha256'] is None:
        parent=json.loads(Path(config['sft_summary']).read_text())
        selected=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=False)
        assert parent['completed_updates']==10000 and selected['step']==parent['selected_step']
        config['initial_checkpoint_sha256']=file_hash(config['initial_checkpoint'])
        config['initial_step']=selected['step']
        config['sft_parent_identity']=parent['identity']
    assert file_hash(config['teacher_checkpoint'])==config['teacher_sha256']
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(config['cpu_threads_per_rank'])
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=90))
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(args.data)/'manifest.json').read_text())
    assert file_hash(Path(args.data)/'dataset.pt')==manifest['sha256']
    dataset=torch.load(Path(args.data)/'dataset.pt',map_location='cpu',weights_only=False)
    assert dataset['identity']==manifest['identity']
    identity=dict(source=source_id(),config=digest(config),data=manifest['sha256'],initial_model=config['initial_checkpoint_sha256'],world=world,cost_version=COST_VERSION)
    idfile=run/'identity.json'
    if idfile.exists():assert json.loads(idfile.read_text())==identity
    elif rank==0:atomic_json(idfile,identity);atomic_json(run/'config.json',config)
    barrier(world)
    torch.manual_seed(config['seed']+rank)
    policy=initialize(config,device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=config['learning_rate'])
    rng=torch.Generator(device=device).manual_seed(config['seed']+100000+rank)
    graph_rng=torch.Generator().manual_seed(config['seed']+200000+rank)
    network=DDP(policy,device_ids=[local],broadcast_buffers=False) if world>1 and device.type=='cuda' else DDP(policy) if world>1 else policy
    start,best,elapsed=restore(policy,optimizer,rng,graph_rng,identity,run,rank,world,device)
    for split in ['train','validation','test']:
        # Smoke overrides may use a prefix; full main uses all declared examples.
        count=config[split+'_graphs'];assert count<=len(dataset['splits'][split]['graphs'])
        dataset['splits'][split]={k:v[:count] for k,v in dataset['splits'][split].items()}
    training=dataset['splits']['train']
    graphs=training['graphs'].to(device);starts=training['starts'].to(device)
    reference=run/'evaluation/validation/summary_step000000.json'
    if not reference.exists():
        baseline=evaluate(policy,dataset['splits']['validation'],config,device,rank,world,run,'validation',0)
        best=dict(cost=baseline['by_budget'][str(config['selection_budget'])]['cost'],step=0)
        if rank==0:atomic_torch(run/'best_model.pt',dict(model=cpu(policy.state_dict()),step=0,best=best,identity=identity))
        barrier(world)
    elif start==0:
        baseline=json.loads(reference.read_text());best=dict(cost=baseline['by_budget'][str(config['selection_budget'])]['cost'],step=0)
        policy.load_state_dict(torch.load(run/'best_model.pt',map_location=device,weights_only=False)['model'])
    initial_elapsed=elapsed;began=time.perf_counter();last=start;durations=[]
    for step in range(start+1,config['updates']+1):
        tick=time.perf_counter();ids=torch.randint(len(graphs),(config['graphs_per_gpu'],),generator=graph_rng)
        selected=ids.to(device)
        trace=collect(policy,graphs[selected],starts[selected],config['horizon'],config['group'],rng,config,device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        collected=time.perf_counter();metrics=update(network,policy,optimizer,trace,config,device,world>1)
        if device.type=='cuda':torch.cuda.synchronize(device)
        trained=time.perf_counter()
        metrics=reduced_metrics(metrics,device,world)
        seconds=max_time(trained-tick,device,world);durations.append(seconds)
        metrics.update(step=step,rollout_seconds=max_time(collected-tick,device,world),
                       update_seconds=max_time(trained-collected,device,world),step_seconds=seconds,
                       global_graphs=world*config['graphs_per_gpu'],global_rollouts=world*config['graphs_per_gpu']*config['group'],
                       global_actions=world*config['graphs_per_gpu']*config['group']*config['horizon'],
                       peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0)
        atomic_torch(run/'sampled_examples'/f'step{step:06d}_rank{rank}.pt',dict(step=step,graph_ids=ids))
        if step in [1,10] or step%config['trace_every']==0:
            atomic_torch(run/'traces'/f'step{step:06d}_rank{rank}.pt',cpu(dict(identity=identity,step=step,graph_ids=ids,trace=trace)))
        del trace
        if rank==0:
            atomic_json(run/'metrics'/f'step{step:06d}.json',metrics)
            atomic_json(run/'status.json',dict(phase='training',step=step,target=config['updates'],best=best,
                        recent_step_seconds=sum(durations[-10:])/len(durations[-10:]),
                        remaining_compute_seconds=(config['updates']-step)*sum(durations[-10:])/len(durations[-10:]),
                        elapsed_training_seconds=initial_elapsed+time.perf_counter()-began))
            print('UPDATE '+json.dumps(metrics),flush=True)
        if step%config['validate_every']==0 or step==config['updates']:
            validation=evaluate(policy,dataset['splits']['validation'],config,device,rank,world,run,'validation',step)
            cost=validation['by_budget'][str(config['selection_budget'])]['cost']
            if cost<best['cost']:
                best=dict(cost=cost,step=step)
                if rank==0:atomic_torch(run/'best_model.pt',dict(model=cpu(policy.state_dict()),step=step,best=best,identity=identity))
            if rank==0:
                comparison=paired_comparison(run,'validation',step,'validation',0,world,config['validation_budgets'])
                atomic_json(run/'evaluation/validation'/f'comparison_step{step:06d}.json',comparison)
                print('VALIDATION '+json.dumps(dict(step=step,cost=cost,best=best,paired=comparison)),flush=True)
            barrier(world)
        elapsed=initial_elapsed+time.perf_counter()-began
        pause=bool(args.max_new_updates and step-start>=args.max_new_updates)
        deadline=(time.perf_counter()-began)>=config['maximum_training_seconds']
        # All ranks must agree whether to stop after a deadline.
        stop=torch.tensor(int(deadline),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        deadline=bool(stop)
        if step%config['checkpoint_every']==0 or step==config['updates'] or pause or deadline:
            save_checkpoint(policy,optimizer,step,best,elapsed,rng,graph_rng,identity,config,run,rank,world,device)
        last=step
        if pause or deadline:break
    if last<config['updates']:
        if rank==0:atomic_json(run/'status.json',dict(phase='paused',step=last,target=config['updates'],best=best,
                            elapsed_training_seconds=elapsed,reason='requested-gate' if args.max_new_updates else 'time-cap'))
        barrier(world)
        if world>1:dist.destroy_process_group()
        return
    if rank==0:atomic_json(run/'status.json',dict(phase='final_evaluation',step=last,best=best))
    chosen=torch.load(run/'best_model.pt',map_location='cpu',weights_only=False);policy.load_state_dict(chosen['model'])
    student=evaluate(policy,dataset['splits']['test'],config,device,rank,world,run,'test_student',chosen['step'],True)
    policy.load_state_dict(initial_model(config,device).state_dict())
    frozen=evaluate(policy,dataset['splits']['test'],config,device,rank,world,run,'test_frozen',0,True)
    policy.load_state_dict(frozen_teacher(config,device).state_dict())
    teacher_test=evaluate(policy,dataset['splits']['test'],config,device,rank,world,run,'test_teacher',config['teacher_step'],True)
    barrier(world)
    if rank==0:
        heuristics=final_heuristics(dataset['splits']['test'],config,run)
        paired=paired_comparison(run,'test_student',chosen['step'],'test_frozen',0,world,config['test_budgets'])
        paired_teacher=paired_comparison(run,'test_student',chosen['step'],'test_teacher',config['teacher_step'],world,config['test_budgets'])
        result=dict(completed_updates=last,selected_step=chosen['step'],identity=identity,student=student,frozen=frozen,teacher_policy=teacher_test,
                    paired=paired,paired_teacher=paired_teacher,heuristics=heuristics['means'],elapsed_training_seconds=elapsed)
        atomic_json(run/'final_summary.json',result);atomic_json(run/'status.json',dict(phase='complete',step=last,best=best))
        print('COMPLETE '+json.dumps(result),flush=True)
    barrier(world)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
