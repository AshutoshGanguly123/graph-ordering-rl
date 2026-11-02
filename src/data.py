"""Read committed policy-teacher shards; graph-uniform state sampling."""
import json
from pathlib import Path
import torch
from common import file_hash

def load(folder,split,limit=None):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());samples=[]
    for item in manifest['splits'][split]:
        path=folder/item['file'];assert file_hash(path)==item['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['identity']==manifest['identity']
        samples.extend(saved['samples'])
        if limit is not None and len(samples)>=limit:break
    if limit is not None:samples=samples[:limit];assert len(samples)==limit
    counts=torch.tensor([len(s['actions']) for s in samples],dtype=torch.int64)
    return dict(graphs=torch.stack([s['graph'] for s in samples]),
                starts=torch.stack([s['start'] for s in samples]).long(),
                teacher_orders=torch.stack([s['final'] for s in samples]).long(),
                teacher_costs=torch.tensor([int(s['costs'][-1]) for s in samples]),
                counts=counts,orders=torch.cat([s['orders'] for s in samples]),
                actions=torch.cat([s['actions'] for s in samples]).long(),
                fractions=torch.cat([s['fractions'] for s in samples]))

def sample(pool,size,rng,device):
    valid=(pool['counts']>0).nonzero().flatten()
    if not len(valid):raise ValueError('No action examples in training data')
    graph_ids=valid[torch.randint(len(valid),(size,),generator=rng,device=device)]
    offsets=pool['counts'].cumsum(0)-pool['counts']
    positions=offsets[graph_ids]+(torch.rand(size,generator=rng,device=device)*pool['counts'][graph_ids]).long()
    return graph_ids,positions

