"""Graph-uniform supervised imitation, durable DDP checkpoints and held-out rollouts."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from common import ROOT,atomic_json,atomic_torch,autocast,digest,file_hash,source_id
from data import load,sample
from environment import reorder
from evaluation import evaluate
from model import SwapPolicy

def checkpoint(run,step,raw,opt,rng,identity,progress,rank,world,device):
    folder=run/'checkpoints'/f'step{step:06d}'
    atomic_torch(folder/f'rank{rank}.pt',dict(identity=identity,step=step,
             sample_rng=rng.get_state(),torch_rng=torch.get_rng_state(),
             cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None))
    if rank==0:atomic_torch(folder/'model.pt',dict(identity=identity,step=step,model=raw.state_dict(),
                              optimizer=opt.state_dict(),progress=progress,
                              best_model=torch.load(run/'best_model.pt',map_location='cpu',weights_only=False)))
    if world>1:dist.barrier()
    if rank==0:
        atomic_json(folder/'committed.json',dict(step=step,identity=identity))
        atomic_json(run/'latest_checkpoint.json',dict(step=step,path=str(folder)))
    if world>1:dist.barrier()

def restore(run,raw,opt,rng,identity,rank,device):
    path=run/'latest_checkpoint.json'
    if not path.exists():return 0,dict(best_calls=None,best_step=0,elapsed_seconds=0.)
    pointer=json.loads(path.read_text());folder=Path(pointer['path'])
    committed=json.loads((folder/'committed.json').read_text())
    assert committed==dict(step=pointer['step'],identity=identity),'Incompatible checkpoint'
    saved=torch.load(folder/'model.pt',map_location=device,weights_only=False)
    state=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    assert saved['identity']==state['identity']==identity and saved['step']==state['step']==pointer['step']
    raw.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    if rank==0:atomic_torch(run/'best_model.pt',saved['best_model'])
    rng.set_state(state['sample_rng'].cpu());torch.set_rng_state(state['torch_rng'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'],device)
    return pointer['step'],saved['progress']

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.json'))
    p.add_argument('--data',required=True);p.add_argument('--run',required=True)
    p.add_argument('--overrides',default='{}');p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--max-new-updates',type=int,default=0)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());config.update(json.loads(a.overrides))
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    torch.set_num_threads(4 if a.device=='cuda' else 1)
    if a.device=='cuda':torch.cuda.set_device(local);device=torch.device('cuda',local)
    else:device=torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=45),
                       **({'device_id':device} if device.type=='cuda' else {}))
    torch.manual_seed(config['seed']);torch.set_float32_matmul_precision('high')
    if config.get('reference_checkpoint'):
        assert file_hash(config['reference_checkpoint'])==config['reference_sha256'],'Frozen teacher changed'
    if config['global_batch']%world:raise ValueError('Global batch must divide across ranks')
    per_gpu=config['global_batch']//world
    manifest=json.loads((Path(a.data)/'manifest.json').read_text())
    ref=config.get('reference_checkpoint')
    identity=dict(source=source_id(),config=digest(config),data=manifest['content_hash'],world=world,
                  reference=file_hash(ref) if ref else None)
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'identity.json').exists():assert json.loads((run/'identity.json').read_text())==identity,'Run identity mismatch'
    if rank==0:
        atomic_json(run/'identity.json',identity);atomic_json(run/'config.json',config)
        atomic_json(run/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                  node=platform.node(),world=world,gpus=os.getenv('CUDA_VISIBLE_DEVICES'),per_gpu_batch=per_gpu))
    if world>1:dist.barrier()
    raw=SwapPolicy(config['side'],config['width'],config['layers'],config['head_width']).to(device)
    network=DDP(raw,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else raw
    opt=torch.optim.Adam(raw.parameters(),lr=config['learning_rate'])
    rng=torch.Generator(device=device).manual_seed(config['seed']+10000+rank*100000)
    start,progress=restore(run,raw,opt,rng,identity,rank,device)
    if (run/'final_summary.json').exists():
        if rank==0:print('ALREADY_COMPLETE',flush=True)
        if world>1:dist.destroy_process_group()
        return
    validation=load(a.data,'validation',config['validation_graphs'])
    train_cpu=load(a.data,'train',config['train_graphs'])
    training={k:v.to(device) for k,v in train_cpu.items()};del train_cpu
    def validation_at(step):
        with torch.random.fork_rng(devices=[local] if device.type=='cuda' else []):
            return evaluate(raw,validation,config,run,'validation',step,rank,world,device,manifest['content_hash'])
    if rank==0:print('START '+json.dumps(dict(start=start,parameters=sum(v.numel() for v in raw.parameters()),
                              identity=identity,config=config)),flush=True)
    if not (run/'latest_checkpoint.json').exists():
        report=validation_at(0);value=report['by_budget'][str(config['selection_budget'])]['student_best_calls']
        progress.update(best_calls=value,best_step=0)
        if rank==0:atomic_torch(run/'best_model.pt',dict(step=0,identity=identity,model=raw.state_dict()))
        checkpoint(run,0,raw,opt,rng,identity,progress,rank,world,device)
    begin=time.perf_counter();elapsed_before=progress['elapsed_seconds'];final_step=start;paused=False
    for step in range(start+1,config['updates']+1):
        tick=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        graph_ids,positions=sample(training,per_gpu,rng,device)
        matrix=reorder(training['graphs'][graph_ids],training['orders'][positions].long())
        opt.zero_grad(set_to_none=True)
        with autocast(device,config):logits=network(matrix,torch.zeros(per_gpu,device=device),training['fractions'][positions])
        loss=F.cross_entropy(logits,training['actions'][positions]);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),config['gradient_clip'])
        finite=torch.tensor(int(bool(torch.isfinite(loss)&torch.isfinite(norm))),device=device)
        if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not bool(finite):raise FloatingPointError('Nonfinite loss/gradients before optimizer update')
        opt.step()
        # Compact raw sampled-example trace; full chronological teacher data is immutable.
        atomic_torch(run/'sampled_examples'/f'rank{rank}'/f'step{step:06d}.pt',
                     dict(step=step,graph_ids=graph_ids.cpu().to(torch.int32),positions=positions.cpu().to(torch.int32)))
        if device.type=='cuda':torch.cuda.synchronize(device)
        values=torch.tensor([float(loss.detach()),float(norm.detach()),time.perf_counter()-tick,
                            torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(values);values/=world
        metrics=dict(step=step,train_ce=float(values[0]),grad_norm=float(values[1]),step_seconds=float(values[2]),
                     peak_vram_mean_bytes=float(values[3]),global_batch=config['global_batch'],
                     elapsed_seconds=elapsed_before+time.perf_counter()-begin)
        if rank==0:
            atomic_json(run/'metrics'/f'step{step:06d}.json',metrics)
            atomic_json(run/'status.json',dict(phase='training',**metrics))
            if step%25==0 or step<=3:print('TRAIN '+json.dumps(metrics),flush=True)
        final_step=step
        if step%config['validate_every']==0 or step==config['updates']:
            report=validation_at(step);value=report['by_budget'][str(config['selection_budget'])]['student_best_calls']
            if value<progress['best_calls']:
                progress.update(best_calls=value,best_step=step)
                if rank==0:atomic_torch(run/'best_model.pt',dict(step=step,identity=identity,model=raw.state_dict()))
        paused=bool(a.max_new_updates and step-start>=a.max_new_updates)
        if step%config['checkpoint_every']==0 or step==config['updates'] or paused:
            progress['elapsed_seconds']=elapsed_before+time.perf_counter()-begin
            checkpoint(run,step,raw,opt,rng,identity,progress,rank,world,device)
        if paused:break
    if paused and final_step<config['updates']:
        if rank==0:atomic_json(run/'status.json',dict(phase='checkpointed_pause',step=final_step,progress=progress))
    else:
        chosen=torch.load(run/'best_model.pt',map_location=device,weights_only=False);raw.load_state_dict(chosen['model'])
        test=load(a.data,'test',config['test_graphs'])
        report=evaluate(raw,test,config,run,'test',chosen['step'],rank,world,device,manifest['content_hash'])
        if rank==0:
            result=dict(completed_updates=final_step,selected_step=chosen['step'],test=report,identity=identity)
            atomic_json(run/'final_summary.json',result);atomic_json(run/'status.json',dict(phase='complete',**result))
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
