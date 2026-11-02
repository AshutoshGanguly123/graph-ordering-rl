"""Fresh, deterministic graph-only pool; no teacher trajectories for training."""
import argparse
import json
from pathlib import Path
import torch
from common import atomic_json, atomic_torch, digest, file_hash
from environment import COST_VERSION, exact_counts, random_orders, reorder, sample_graphs, recursive_reference

def build(config, output):
    torch.set_num_threads(1)
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    spec={k:config[k] for k in ['side','train_graphs','validation_graphs','test_graphs','train_seed','validation_seed','test_seed']}
    identity=digest(dict(spec=spec, environment=file_hash(Path(__file__).with_name('environment.py'))))
    manifest=output/'manifest.json'; target=output/'dataset.pt'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        assert saved['identity']==identity and file_hash(target)==saved['sha256']
        return saved
    pool={}
    for split in ['train','validation','test']:
        graphs=[];starts=[]
        for i in range(config[split+'_graphs']):
            rng=torch.Generator().manual_seed(config[split+'_seed']+i)
            graph=sample_graphs(1,config['side'],rng,'cpu')[0]
            start=random_orders(1,config['side'],rng,'cpu')[0]
            assert torch.equal(graph,graph.T) and not bool(graph.diag().any())
            if i<10:
                matrix=reorder(graph[None],start[None])[0]
                calls,decisions=exact_counts(matrix[None])
                assert (int(calls[0]),int(decisions[0]))==recursive_reference(matrix)
            graphs.append(graph);starts.append(start.to(torch.int16))
        graphs=torch.stack(graphs); starts=torch.stack(starts)
        initial=exact_counts(reorder(graphs,starts.long()))[0]
        pool[split]=dict(graphs=graphs,starts=starts,initial=initial,
                         seeds=torch.arange(config[split+'_graphs'])+config[split+'_seed'])
    atomic_torch(target,dict(identity=identity,splits=pool,cost_version=COST_VERSION))
    result=dict(identity=identity,sha256=file_hash(target),spec=spec,
                graphs={k:len(v['graphs']) for k,v in pool.items()},cost_version=COST_VERSION)
    atomic_json(manifest,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(json.loads(Path(a.config).read_text()),a.output),indent=2))
