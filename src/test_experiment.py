"""Objective/compiler tests and a tiny full learned-teacher -> SFT -> RL cycle."""
import copy,json,os,subprocess,sys,tempfile
from pathlib import Path
import torch
from common import ROOT,atomic_json,atomic_torch,file_hash
from model import SwapPolicy
from environment import exact_counts,random_orders,recursive_reference,reorder,sample_graphs,legal_pairs
from teacher_data import build,compile_sample,audit_sample
from rl_train import collect,reward_and_advantage,save_checkpoint,restore,update
from rl_data import build as build_rl
from data import load

def run(command):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
    if result.returncode:raise AssertionError(result.stdout[-5000:]+result.stderr[-5000:])

def tests():
    torch.set_num_threads(1);torch.manual_seed(123);device=torch.device('cpu')
    for side in [4,8,64]:
        rng=torch.Generator().manual_seed(side+123)
        for graph in sample_graphs(2,side,rng,device):
            calls,decisions=exact_counts(graph[None]);assert (int(calls[0]),int(decisions[0]))==recursive_reference(graph)
    initial=torch.tensor([100]*4)
    costs=torch.tensor([[90,100,100,100],[95,95,100,100],[85,98,100,100],[92,99,100,100]])
    best=torch.cat([initial[None],costs]).cummin(0).values[1:]
    rewards,returns,advantages=reward_and_advantage(initial,costs,best,4)
    assert torch.allclose(rewards.sum(0),(initial-best[-1]).float()/initial)
    assert not torch.equal(advantages[0],advantages[-1]);assert abs(float(rewards[1,0])+.05)<1e-7
    assert torch.allclose(advantages.sum(-1),torch.zeros(4),atol=1e-7)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);teacher=folder/'teacher.pt'
        atomic_torch(teacher,dict(step=950,model=SwapPolicy(4,16,3,16).state_dict()))
        config=json.loads((ROOT/'config.json').read_text())
        config.update(side=4,width=16,head_width=16,teacher_checkpoint=str(teacher),teacher_sha256=file_hash(teacher),
                    teacher_group=4,teacher_horizon=8,teacher_graph_batch=2,train_graphs=8,validation_graphs=4,test_graphs=4,
                    audit_count=8,reference_checkpoint=str(teacher),reference_sha256=file_hash(teacher),
                    global_batch=8,updates=2,validate_every=1,checkpoint_every=1,evaluation_group=4,
                    evaluation_budgets=[4,8],selection_budget=8,evaluation_graph_batch=2,ce_batch=8,precision='fp32')
        # Real checkpoint -> sampled actions -> audited chronological winning prefixes.
        build(config,folder/'sft_data',device,audit_only=True)
        assert not (folder/'sft_data/manifest.json').exists()
        build(config,folder/'sft_data',device)
        manifest=(folder/'sft_data/manifest.json').read_bytes();build(config,folder/'sft_data',device)
        assert (folder/'sft_data/manifest.json').read_bytes()==manifest
        pool=load(folder/'sft_data','train');assert len(pool['graphs'])==8
        assert int(pool['counts'].sum())>0
        # SFT is freshly initialized and survives an exact-identity checkpoint resume.
        atomic_json(folder/'sft_config.json',config)
        command=[sys.executable,str(ROOT/'sft_train.py'),'--device','cpu','--config',str(folder/'sft_config.json'),
                 '--data',str(folder/'sft_data'),'--run',str(folder/'sft_run')]
        run(command+['--max-new-updates','1']);run(command)
        summary=json.loads((folder/'sft_run/final_summary.json').read_text());assert summary['completed_updates']==2
        selected=torch.load(folder/'sft_run/best_model.pt',map_location='cpu',weights_only=False)
        assert not torch.equal(selected['model']['input.weight'],torch.load(teacher,weights_only=False)['model']['input.weight'])
        rl=json.loads((ROOT/'rl_config.json').read_text())
        rl.update(side=4,width=16,head_width=16,initial_checkpoint=str(folder/'sft_run/best_model.pt'),
                  initial_checkpoint_sha256=file_hash(folder/'sft_run/best_model.pt'),initial_step=selected['step'],
                  teacher_checkpoint=str(teacher),teacher_sha256=file_hash(teacher),
                  train_graphs=8,validation_graphs=4,test_graphs=4,graphs_per_gpu=2,horizon=8,updates=2,
                  microbatch=8,precision='fp32',cpu_threads_per_rank=1,validate_every=1,checkpoint_every=1,trace_every=1,
                  evaluation_group=4,evaluation_graph_batch=2,validation_budgets=[4,8],selection_budget=8,
                  test_group=4,test_graph_batch=2,test_budgets=[4,8])
        build_rl(rl,folder/'rl_data');atomic_json(folder/'rl_config.json',rl)
        command=[sys.executable,str(ROOT/'rl_train.py'),'--config',str(folder/'rl_config.json'),
                 '--data',str(folder/'rl_data'),'--run',str(folder/'rl_run')]
        run(command+['--max-new-updates','1']);run(command)
        final=json.loads((folder/'rl_run/final_summary.json').read_text())
        assert final['completed_updates']==2 and final['teacher_policy']['step']==950 and 'paired_teacher' in final
        # Exact stochastic optimizer/RNG recovery, not just matching checkpoint steps.
        policy=SwapPolicy(4,16,3,16);optimizer=torch.optim.Adam(policy.parameters(),lr=3e-5)
        gen=torch.Generator().manual_seed(18);grng=torch.Generator().manual_seed(19)
        graphs=sample_graphs(2,4,grng,device);starts=random_orders(2,4,grng,device)
        small=dict(precision='fp32',microbatch=7,gradient_clip=1.)
        trace=collect(policy,graphs,starts,8,4,gen,small,device)
        stats=update(policy,policy,optimizer,trace,small,device);assert stats['recompute_logp_abs_delta']<1e-5
        checkpoint=folder/'exact_resume';identity=dict(test='exact')
        save_checkpoint(policy,optimizer,1,dict(cost=1.,step=0),1.,gen,grng,identity,small,checkpoint,0,1,device)
        next1=collect(policy,graphs,starts,8,4,gen,small,device);update(policy,policy,optimizer,next1,small,device)
        wanted=copy.deepcopy(policy.state_dict())
        again=SwapPolicy(4,16,3,16);opt2=torch.optim.Adam(again.parameters(),lr=3e-5);g2=torch.Generator();g3=torch.Generator()
        restore(again,opt2,g2,g3,identity,checkpoint,0,1,device)
        next2=collect(again,graphs,starts,8,4,g2,small,device)
        assert torch.equal(next1['actions'],next2['actions'])
        update(again,again,opt2,next2,small,device)
        assert all(torch.equal(wanted[k],again.state_dict()[k]) for k in wanted)
    return dict(passed=True,checks=['literal recursive cost at 3 sizes','dense reward telescoping and per-step LOO',
                'learned teacher best-prefix compilation, legality and edge preservation',
                'audited shard generation and deterministic committed-shard resume',
                'SFT from scratch, checkpointed resume and held-out selection',
                'RL initialization from selected student, pause/resume and final teacher comparison',
                'exact model/Adam/sampling RNG recovery and logp replay'])

if __name__=='__main__':print(json.dumps(tests(),indent=2))
