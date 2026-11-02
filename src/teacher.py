"""One deterministic preorder trace implementing balanced spectral partitions."""
import torch
from environment import exact_counts,legal_pairs,random_orders,recursive_reference,reorder,sample_graphs

def split_ids(affinity,ids):
    w=affinity[ids][:,ids].clone();w.fill_diagonal_(0)
    if not bool(w.any()):return ids
    _,vectors=torch.linalg.eigh(torch.diag(w.sum(-1))-w)
    vector=vectors[:,1]
    if vector[int(vector.abs().argmax())]<0:vector=-vector
    return [ids[i] for i in vector.argsort(stable=True).tolist()]

def reference_order(affinity,ids):
    if len(ids)<2:return ids
    ranked=split_ids(affinity,ids);mid=len(ids)//2
    return reference_order(affinity,ranked[:mid])+reference_order(affinity,ranked[mid:])

def generate(seed,side=64,audit=False):
    rng=torch.Generator().manual_seed(seed)
    graph=sample_graphs(1,side,rng,'cpu')[0]
    start=random_orders(1,side,rng,'cpu')[0]
    a=reorder(graph[None],start[None])[0];b=a[:side,side:].double()
    current=list(range(2*side));orders=[];actions=[];swaps=[];splits=[]
    pairs=legal_pairs(side,'cpu');lookup={tuple(v):i for i,v in enumerate(pairs.tolist())}
    def visit(ids,offset,affinity,partition,depth):
        if len(ids)<2:return ids
        ranked=split_ids(affinity,ids);mid=len(ids)//2;left=set(ranked[:mid])
        splits.append(dict(partition=partition,offset=offset,depth=depth,
                           ids=ids,left=ranked[:mid],right=ranked[mid:]))
        wrong_l=[p for p in range(offset,offset+mid) if current[p]%side not in left]
        wrong_r=[p for p in range(offset+mid,offset+len(ids)) if current[p]%side in left]
        assert len(wrong_l)==len(wrong_r)
        for i,j in zip(wrong_l,wrong_r):
            orders.append(current.copy());actions.append(lookup[(i,j)]);swaps.append((i,j))
            current[i],current[j]=current[j],current[i]
        return visit(ranked[:mid],offset,affinity,partition,depth+1)+visit(ranked[mid:],offset+mid,affinity,partition,depth+1)
    u=visit(list(range(side)),0,b@b.T,'U',0)
    v=visit(list(range(side)),side,b.T@b,'V',0)
    assert current==u+[x+side for x in v]
    assert u==reference_order(b@b.T,list(range(side)))
    assert v==reference_order(b.T@b,list(range(side)))
    final=start[torch.tensor(current)]
    orders=start[torch.tensor(orders,dtype=torch.int64).reshape(-1,2*side)]
    all_orders=torch.cat([orders,final[None]])
    assert len(actions)<=side*int(side.bit_length()-1)
    for t,(i,j) in enumerate(swaps):
        assert (i<side)==(j<side)
        expected=all_orders[t].clone();expected[i],expected[j]=all_orders[t,j],all_orders[t,i]
        assert torch.equal(expected,all_orders[t+1])
    assert torch.equal(all_orders.sort(-1).values,torch.arange(2*side)[None].expand_as(all_orders))
    matrices=reorder(graph[None].expand(len(all_orders),-1,-1),all_orders)
    calls,decisions=exact_counts(matrices)
    indices=range(len(all_orders)) if audit and side<=8 else sorted(set([0,len(actions)//2,len(actions)]))
    for i in indices:assert (int(calls[i]),int(decisions[i]))==recursive_reference(matrices[i])
    if audit:
        inverses=all_orders.argsort(-1)
        assert torch.equal(reorder(matrices,inverses),graph[None].expand_as(matrices))
    return dict(seed=seed,graph=graph,start=start.to(torch.int16),orders=orders.to(torch.int16),
                final=final.to(torch.int16),actions=torch.tensor(actions,dtype=torch.int16),
                swaps=torch.tensor(swaps,dtype=torch.int16).reshape(-1,2),costs=calls.to(torch.int32),
                fractions=calls[:-1].cummin(0).values.float()/calls[0],splits=splits)
