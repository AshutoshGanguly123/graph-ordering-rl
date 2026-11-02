"""Exact graph-uniform CE, cached baselines, and autonomous budget prefixes."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from common import atomic_json,atomic_torch,autocast,digest,file_hash
from environment import exact_counts,reorder,swap_permutations
from model import SwapPolicy

@torch.no_grad()
def rollout(policy,graphs,starts,config,rng,device,mode='student'):
    group=config['evaluation_group'];budgets=config['evaluation_budgets'];horizon=max(budgets)
    base=graphs.repeat_interleave(group,0);order=starts.repeat_interleave(group,0).clone()
    initial,_=exact_counts(reorder(base,order));best=initial.clone();best_order=order.clone()
    actions=[];proposed=[];incumbents=[];logps=[];prefix_orders={}
    for t in range(horizon):
        if mode=='random':logits=torch.zeros(len(order),len(policy.pairs),device=device)
        else:
            progress=torch.full((len(order),),(horizon-t)/horizon if mode=='previous' and config.get('reference_countdown',True) else 0.,device=device)
            with autocast(device,config):logits=policy(reorder(base,order),progress,best.float()/initial)
        noise=torch.rand(logits.shape,generator=rng,device=device).clamp_(1e-7,1-1e-7)
        action=(logits-noise.log().neg().log()).argmax(-1)
        lp=logits.log_softmax(-1).gather(1,action[:,None]).squeeze(1)
        order=swap_permutations(order,policy.pairs[action])
        cost,_=exact_counts(reorder(base,order));improved=cost<best
        best_order=torch.where(improved[:,None],order,best_order);best=torch.minimum(best,cost)
        actions.append(action.to(torch.int16));proposed.append(cost.to(torch.int32))
        incumbents.append(best.to(torch.int32));logps.append(lp)
        if t+1 in budgets:prefix_orders[t+1]=best_order.to(torch.int16).clone()
    return dict(starts=starts.to(torch.int16),initial=initial,actions=torch.stack(actions),
                costs=torch.stack(proposed),best_costs=torch.stack(incumbents),
                logps=torch.stack(logps),prefix_orders=prefix_orders,group=group)

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    return value

@torch.no_grad()
def validation_ce(policy,pool,config,ids,device):
    counts=pool['counts'];offset=counts.cumsum(0)-counts
    valid=[i for i in ids if int(counts[i])>0]
    state_ids=torch.cat([torch.arange(int(offset[i]),int(offset[i]+counts[i])) for i in valid]) if valid else torch.empty(0,dtype=torch.long)
    graph_ids=torch.cat([torch.full((int(counts[i]),),i) for i in valid]) if valid else torch.empty(0,dtype=torch.long)
    total=torch.zeros((),device=device,dtype=torch.float64)
    for lo in range(0,len(state_ids),config['ce_batch']):
        pos=state_ids[lo:lo+config['ce_batch']];gi=graph_ids[lo:lo+config['ce_batch']]
        matrix=reorder(pool['graphs'][gi].to(device),pool['orders'][pos].long().to(device))
        fraction=pool['fractions'][pos].to(device)
        with autocast(device,config):logits=policy(matrix,torch.zeros(len(pos),device=device),fraction)
        losses=F.cross_entropy(logits,pool['actions'][pos].to(device),reduction='none')
        total+=(losses.double()/counts[gi].to(device)).sum()
    return total,len(valid)

@torch.no_grad()
def evaluate(policy,pool,config,run,split,step,rank,world,device,data_identity):
    policy.eval();run=Path(run);ids=list(range(rank,len(pool['graphs']),world))
    budgets=config['evaluation_budgets'];group=config['evaluation_group']
    previous=None;ref=config.get('reference_checkpoint')
    reference_hash=file_hash(ref) if ref else None
    cache_identity=digest(dict(data=data_identity,split=split,budgets=budgets,group=group,
                               world=world,chunk=config['evaluation_graph_batch'],reference=reference_hash))
    folder=run/'evaluation'/split;folder.mkdir(parents=True,exist_ok=True)
    if rank==0:atomic_json(run/'status.json',dict(phase='evaluating',split=split,step=step))
    ce_sum,ce_n=validation_ce(policy,pool,config,ids,device)
    totals=torch.zeros(len(budgets),9,dtype=torch.float64,device=device);records=[]
    chunk=config['evaluation_graph_batch'];base_seed=config[split+'_seed']
    for lo in range(0,len(ids),chunk):
        block=ids[lo:lo+chunk];graphs=pool['graphs'][block].to(device);starts=pool['starts'][block].to(device)
        cache=folder/f'baselines_rank{rank}_block{lo//chunk:04d}.pt'
        if cache.exists():
            saved=torch.load(cache,map_location='cpu',weights_only=False)
            assert saved['identity']==cache_identity and saved['ids']==block
        else:
            if ref and previous is None:
                previous=SwapPolicy(config['side'],config['width'],config['layers'],config['head_width']).to(device)
                chosen=torch.load(ref,map_location='cpu',weights_only=False);previous.load_state_dict(chosen['model']);previous.eval()
            reference_rng=torch.Generator(device=device).manual_seed(base_seed+100000+rank*1000+lo)
            random_trace=rollout(policy,graphs,starts,config,reference_rng,device,'random')
            previous_trace=None
            if previous is not None:
                reference_rng.manual_seed(base_seed+200000+rank*1000+lo)
                previous_trace=rollout(previous,graphs,starts,config,reference_rng,device,'previous')
            saved=dict(identity=cache_identity,ids=block,random=cpu(random_trace),previous=cpu(previous_trace))
            atomic_torch(cache,saved)
        rng=torch.Generator(device=device).manual_seed(base_seed+300000+rank*1000+lo)
        trace=rollout(policy,graphs,starts,config,rng,device)
        atomic_torch(folder/f'step{step:06d}_rank{rank}_block{lo//chunk:04d}.pt',dict(identity=cache_identity,ids=block,trace=cpu(trace)))
        teacher=pool['teacher_costs'][block].double().to(device)
        for bi,budget in enumerate(budgets):
            costs=trace['best_costs'][budget-1].reshape(-1,group).double()
            best=costs.min(-1).values;single=costs[:,0]
            random=saved['random']['best_costs'][budget-1].reshape(-1,group).min(-1).values.double().to(device)
            old=(saved['previous']['best_costs'][budget-1].reshape(-1,group).min(-1).values.double().to(device)
                 if saved['previous'] is not None else torch.zeros_like(best))
            gap=(best-teacher)/teacher
            initial=trace['initial'].reshape(-1,group)[:,0].double()
            totals[bi]+=torch.stack([best.sum(),single.sum(),random.sum(),old.sum(),teacher.sum(),
                                  gap.sum(),gap.square().sum(),(best<teacher).double().sum(),initial.sum()])
            records.extend(dict(graph=gi,budget=budget,student_best=int(best[j]),student_single=int(single[j]),
                               random_best=int(random[j]),previous_best=int(old[j]) if ref else None,
                               teacher=int(teacher[j]),initial=int(initial[j])) for j,gi in enumerate(block))
    extra=torch.tensor([float(ce_sum),ce_n,len(ids)],dtype=torch.float64,device=device)
    if world>1:dist.all_reduce(totals);dist.all_reduce(extra)
    count=int(extra[2]);report=dict(split=split,step=step,graphs=count,validation_ce=float(extra[0]/extra[1]) if extra[1]>0 else None,
                                 ce_graphs=int(extra[1]),by_budget={})
    for bi,budget in enumerate(budgets):
        values=totals[bi]/count;gap=float(values[5]);second=float(values[6])
        report['by_budget'][str(budget)]=dict(student_best_calls=float(values[0]),student_single_calls=float(values[1]),
             random_best_calls=float(values[2]),previous_best_calls=float(values[3]) if ref else None,
             teacher_calls=float(values[4]),paired_teacher_gap=gap,
             paired_teacher_gap_se=(max(0.,second-gap*gap)/max(1,count-1))**.5,
             beats_teacher_fraction=float(values[7]),initial_calls=float(values[8]))
    atomic_json(folder/f'per_graph_step{step:06d}_rank{rank}.json',records)
    if rank==0:
        atomic_json(folder/f'summary_step{step:06d}.json',report)
        print('EVALUATION '+json.dumps(report),flush=True)
    policy.train();return report
