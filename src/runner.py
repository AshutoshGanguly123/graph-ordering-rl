"""Finite staged runner; no background monitoring or scheduling service."""
import json,os,subprocess,time
from pathlib import Path
from common import ROOT,atomic_json,file_hash
from launch import PYTHON
from resources import inspect

def phase(name,command,gpu=False,completion=None):
    path=ROOT/'artifacts'/f'{name}.json'
    if path.exists() and json.loads(path.read_text()).get('exit')==0:
        if completion is None or completion():return
    env=os.environ.copy();resources=None
    if gpu:
        resources=inspect()
        env.update(CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c in resources['selected']),OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    record=dict(phase=name,status='running',command=command,resources=resources,started=time.time())
    atomic_json(path,record);atomic_json(ROOT/'artifacts/pipeline.json',record)
    with (ROOT/'artifacts'/f'{name}.log').open('a') as log:
        result=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record.update(exit=result.returncode,status='complete' if result.returncode==0 else 'failed',finished=time.time())
    atomic_json(path,record);atomic_json(ROOT/'artifacts/pipeline.json',record)
    if result.returncode:raise RuntimeError(f'{name} failed; see artifacts/{name}.log')

def torchrun(script,args):
    return [str(Path(PYTHON).with_name('torchrun')),'--standalone','--nnodes=1','--nproc-per-node=4','--',str(ROOT/script)]+args

def main():
    try:
        deployed=json.loads((ROOT/'deployment.json').read_text())
        assert all(file_hash(ROOT/name)==value for name,value in deployed['sources'].items())
        phase('cpu_tests',[PYTHON,str(ROOT/'test_experiment.py')])
        tiny=dict(train_graphs=16,validation_graphs=8,test_graphs=8,audit_count=16,teacher_group=4,teacher_horizon=8,teacher_graph_batch=2)
        phase('teacher_smoke',torchrun('teacher_data.py',['--folder',str(ROOT/'data/smoke'),'--overrides',json.dumps(tiny)]),True)
        tiny_sft=dict(train_graphs=16,validation_graphs=8,test_graphs=8,updates=2,global_batch=32,validate_every=1,checkpoint_every=1,
                      evaluation_group=4,evaluation_budgets=[4,8],evaluation_graph_batch=2,selection_budget=8,ce_batch=16)
        phase('sft_smoke',torchrun('sft_train.py',['--data',str(ROOT/'data/smoke'),'--run',str(ROOT/'runs/sft_smoke'),
                    '--overrides',json.dumps(tiny_sft)]),True)
        # torch checkpoint inspection is performed here with the already loaded runtime.
        import torch
        selected=torch.load(ROOT/'runs/sft_smoke/best_model.pt',map_location='cpu',weights_only=False)
        tiny_rl=dict(train_graphs=8,validation_graphs=8,test_graphs=8,updates=2,graphs_per_gpu=2,group=4,horizon=8,microbatch=16,
                     validate_every=1,checkpoint_every=1,trace_every=1,evaluation_group=4,evaluation_graph_batch=2,
                     validation_budgets=[4,8],selection_budget=8,test_group=4,test_graph_batch=2,test_budgets=[4,8],
                     initial_checkpoint=str(ROOT/'runs/sft_smoke/best_model.pt'),initial_checkpoint_sha256=file_hash(ROOT/'runs/sft_smoke/best_model.pt'),initial_step=selected['step'])
        atomic_json(ROOT/'artifacts/smoke_rl_config.json',dict(json.loads((ROOT/'rl_config.json').read_text()),**tiny_rl))
        phase('rl_smoke_data',[PYTHON,str(ROOT/'rl_data.py'),'--config',str(ROOT/'artifacts/smoke_rl_config.json'),'--output',str(ROOT/'data/rl_smoke')])
        phase('rl_smoke',torchrun('rl_train.py',['--config',str(ROOT/'artifacts/smoke_rl_config.json'),'--data',str(ROOT/'data/rl_smoke'),'--run',str(ROOT/'runs/rl_smoke')]),True)
        phase('teacher_audit',torchrun('teacher_data.py',['--folder',str(ROOT/'data/sft'),'--audit-only']),True)
        phase('teacher_data',torchrun('teacher_data.py',['--folder',str(ROOT/'data/sft')]),True)
        sft=['--data',str(ROOT/'data/sft'),'--run',str(ROOT/'runs/sft')]
        phase('sft_profile',torchrun('sft_train.py',sft+['--max-new-updates','10']),True)
        phase('sft_train',torchrun('sft_train.py',sft),True,lambda:(ROOT/'runs/sft/final_summary.json').exists())
        phase('rl_data',[PYTHON,str(ROOT/'rl_data.py'),'--config',str(ROOT/'rl_config.json'),'--output',str(ROOT/'data/rl')])
        rl=['--config',str(ROOT/'rl_config.json'),'--data',str(ROOT/'data/rl'),'--run',str(ROOT/'runs/rl')]
        phase('rl_profile',torchrun('rl_train.py',rl+['--max-new-updates','10']),True)
        phase('rl_train',torchrun('rl_train.py',rl),True,lambda:(ROOT/'runs/rl/final_summary.json').exists())
        if (ROOT/'runs/rl/final_summary.json').exists():
            atomic_json(ROOT/'artifacts/pipeline.json',dict(phase='complete',status='complete',finished=time.time()))
        else:atomic_json(ROOT/'artifacts/pipeline.json',dict(phase='paused',status='resumable',finished=time.time()))
    except Exception as error:
        atomic_json(ROOT/'artifacts/pipeline_error.json',dict(error=str(error),finished=time.time()));raise

if __name__=='__main__':main()
