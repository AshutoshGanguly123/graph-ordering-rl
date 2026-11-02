"""Sample the immutable learned teacher, compile one winning chronological path."""
import argparse
from datetime import timedelta
import json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from common import ROOT,atomic_json,atomic_torch,cpu,digest,file_hash,frozen_teacher,source_id
from environment import exact_counts,legal_pairs,random_orders,recursive_reference,reorder,sample_graphs
from rl_train import collect

def compile_sample(graph,start,trace,index,group,pairs,seed):
    values=trace['best_costs'][-1,index*group:(index+1)*group]
    winner=int(values.argmin());column=index*group+winner
    initial=int(trace['initial'][column]);target=int(values[winner])
    if target==initial:end=0
    else:end=int((trace['costs'][:,column]==target).nonzero()[0])+1
    actions=trace['actions'][:end,column].to(torch.int16).clone()
    curve=torch.cat([torch.tensor([initial],dtype=torch.int32),trace['costs'][:end,column].to(torch.int32)])
    current=start.numpy().astype(np.int16,copy=True)
    orders=np.empty((end,len(start)),dtype=np.int16)
    lookup=pairs.numpy()
    for t,action in enumerate(actions.tolist()):
        orders[t]=current;i,j=lookup[action]
        assert (i<len(start)//2)==(j<len(start)//2)
        current[i],current[j]=current[j],current[i]
    final=torch.from_numpy(current.copy())
    assert torch.equal(final,trace['best_order'][column].short())
    assert int(curve[-1])==target and int(curve.min())==target
    if end:assert bool((curve[:-1]>target).all()),'Endpoint must be first occurrence of final best'
    return dict(seed=seed,graph=graph,start=start.short(),orders=torch.from_numpy(orders),actions=actions,
                final=final,costs=curve,fractions=curve[:-1].cummin(0).values.float()/initial,
                winner_rollout=winner,best_step=end,teacher_best_cost=target)

def audit_sample(sample,device):
    n=len(sample['start']);side=n//2;orders=torch.cat([sample['orders'],sample['final'][None]]).long()
    assert torch.equal(orders.sort(-1).values,torch.arange(n)[None].expand_as(orders))
    assert bool((orders[:,:side]<side).all()) and bool((orders[:,side:]>=side).all())
    assert torch.equal(orders[0],sample['start'].long())
    graph=sample['graph'].to(device)
    matrices=reorder(graph[None].expand(len(orders),-1,-1),orders.to(device))
    calls,decisions=exact_counts(matrices)
    assert torch.equal(calls.cpu().int(),sample['costs'])
    final=matrices[-1]
    assert torch.equal(reorder(final[None],orders[-1].argsort()[None].to(device))[0],graph)
    for t in sorted(set([0,len(orders)//2,len(orders)-1])):
        assert (int(calls[t]),int(decisions[t]))==recursive_reference(matrices[t].cpu())
    pairs=legal_pairs(side,'cpu')
    for t,action in enumerate(sample['actions'].long()):
        expected=orders[t].clone();i,j=pairs[action]
        expected[i],expected[j]=orders[t,j],orders[t,i]
        assert torch.equal(expected,orders[t+1])
    return dict(seed=sample['seed'],states=len(sample['actions']),initial=int(sample['costs'][0]),
                best=int(sample['costs'][-1]),winner=sample['winner_rollout'],passed=True)

def task_list(config):
    tasks=[]
    for split_index,split in enumerate(['train','validation','test']):
        for lo in range(0,config[split+'_graphs'],config['teacher_graph_batch']):
            seeds=list(range(config[split+'_seed']+lo,config[split+'_seed']+min(lo+config['teacher_graph_batch'],config[split+'_graphs'])))
            tasks.append(dict(split=split,lo=lo,seeds=seeds,sampler_seed=config['teacher_seed']+split_index*1000000+lo,
                              file=f'{split}/shard{lo:06d}.pt'))
    return tasks

def build(config,folder,device,rank=0,world=1,audit_only=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    spec={k:config[k] for k in ['side','width','layers','head_width','teacher_sha256','teacher_step','teacher_group','teacher_horizon',
                              'teacher_graph_batch','teacher_seed','train_graphs','validation_graphs','test_graphs','train_seed','validation_seed','test_seed']}
    identity=digest(dict(spec=spec,source=source_id()))
    marker=folder/'identity.json'
    if marker.exists():assert json.loads(marker.read_text())['identity']==identity
    if rank==0:atomic_json(marker,dict(identity=identity,spec=spec))
    if world>1:dist.barrier()
    policy=frozen_teacher(config,device);pairs=legal_pairs(config['side'],'cpu')
    tasks=task_list(config);own=[task for i,task in enumerate(tasks) if i%world==rank]
    if audit_only:own=[t for t in own if t['split']=='train' and t['lo']<min(config['audit_count'],config['train_graphs'])]
    began=time.perf_counter();committed=0
    for task in own:
        target=folder/task['file'];meta=Path(str(target)+'.json')
        if target.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['identity']==identity and saved['seeds']==task['seeds']
            assert file_hash(target)==saved['sha256'] and file_hash(folder/saved['raw_file'])==saved['raw_sha256']
        else:
            graphs=[];starts=[]
            for seed in task['seeds']:
                rng=torch.Generator().manual_seed(seed)
                graphs.append(sample_graphs(1,config['side'],rng,'cpu')[0])
                starts.append(random_orders(1,config['side'],rng,'cpu')[0])
            graphs=torch.stack(graphs);starts=torch.stack(starts)
            rng=torch.Generator(device=device).manual_seed(task['sampler_seed'])
            trace=collect(policy,graphs.to(device),starts.to(device),config['teacher_horizon'],config['teacher_group'],rng,config,device,capture=False)
            trace=cpu(trace)
            samples=[compile_sample(graphs[i],starts[i],trace,i,config['teacher_group'],pairs,seed) for i,seed in enumerate(task['seeds'])]
            audited=[]
            for i,sample in enumerate(samples):
                if task['split']=='train' and task['lo']+i<config['audit_count']:audited.append(audit_sample(sample,device))
                else:
                    observed=exact_counts(reorder(sample['graph'][None],sample['final'].long()[None]))[0][0]
                    assert int(observed)==sample['teacher_best_cost']
            raw_file=f"raw/{task['split']}_block{task['lo']:06d}.pt"
            atomic_torch(folder/raw_file,dict(identity=identity,seeds=task['seeds'],sampler_seed=task['sampler_seed'],teacher_sha256=config['teacher_sha256'],trace=trace))
            atomic_torch(target,dict(identity=identity,seeds=task['seeds'],samples=samples))
            saved=dict(identity=identity,seeds=task['seeds'],sha256=file_hash(target),graphs=len(samples),
                       states=sum(len(s['actions']) for s in samples),raw_file=raw_file,
                       raw_sha256=file_hash(folder/raw_file),audits=audited,
                       zero_actions=sum(len(s['actions'])==0 for s in samples))
            atomic_json(meta,saved)
        committed+=len(task['seeds'])
        atomic_json(folder/f'progress_rank{rank}.json',dict(committed_graphs=committed,total_graphs=sum(len(t['seeds']) for t in own),
                    seconds=time.perf_counter()-began,last=task['file'],audit_only=audit_only))
        if committed%64==0 or committed==sum(len(t['seeds']) for t in own):print(f'TEACHER_RANK {rank} GRAPHS {committed} / {sum(len(t["seeds"]) for t in own)}',flush=True)
    if world>1:dist.barrier()
    if rank==0:
        audits=[]
        for task in tasks:
            if task['split']=='train' and task['lo']<min(config['audit_count'],config['train_graphs']):
                audits+=json.loads(Path(str(folder/task['file'])+'.json').read_text())['audits']
        assert len(audits)==min(config['audit_count'],config['train_graphs']) and all(a['passed'] for a in audits)
        atomic_json(folder/'audit.json',dict(identity=identity,passed=True,graphs=len(audits),records=audits))
        if not audit_only:
            splits={s:[] for s in ['train','validation','test']}
            for task in tasks:
                item=json.loads(Path(str(folder/task['file'])+'.json').read_text())
                assert item['identity']==identity and file_hash(folder/task['file'])==item['sha256']
                splits[task['split']].append(dict(file=task['file'],**item))
            for split,entries in splits.items():assert sum(x['graphs'] for x in entries)==config[split+'_graphs']
            atomic_json(folder/'manifest.json',dict(identity=identity,spec=spec,splits=splits,
                        content_hash=digest({s:[x['sha256'] for x in es] for s,es in splits.items()}),
                        total_states=sum(e['states'] for es in splits.values() for e in es),
                        zero_actions=sum(e['zero_actions'] for es in splits.values() for e in es)))
            print('TEACHER_DATA_COMPLETE',flush=True)
    if world>1:dist.barrier()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.json'));p.add_argument('--folder',required=True)
    p.add_argument('--overrides',default='{}');p.add_argument('--audit-only',action='store_true');p.add_argument('--device',default='cuda')
    a=p.parse_args();config=json.loads(Path(a.config).read_text());config.update(json.loads(a.overrides))
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    device=torch.device('cuda',local) if a.device=='cuda' else torch.device('cpu');torch.set_num_threads(4 if a.device=='cuda' else 1)
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=90))
    build(config,a.folder,device,rank,world,a.audit_only)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
