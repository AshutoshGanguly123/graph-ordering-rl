"""Graph-preserving swap environment and exact full-width row-tree counts."""
import torch

COST_VERSION = 'full-width-midpoint-roots-and-edge-leaves-v1'

def exact_counts(adjacency):
    """Return entered recursive calls and child decisions; no neural prediction.

    Every row enters a root even when empty. Only occupied children recurse.
    Full symmetric adjacency counts each undirected edge in both directions.
    This is the uncompressed section-2.1 tree convention, not optimized latency.
    """
    n = adjacency.shape[-1]
    if n < 2 or n & (n-1) or adjacency.shape[-2] != n:
        raise ValueError('Require square power-of-two adjacency with n >= 2')
    bits = adjacency.bool()
    internal_nonroot = torch.zeros(bits.shape[0], device=bits.device, dtype=torch.int64)
    width = 2
    while width < n:
        occupied = bits.reshape(-1, n, n//width, width).any(dim=-1)
        internal_nonroot += occupied.sum(dim=(1, 2))
        width *= 2
    calls = n + bits.sum(dim=(1, 2)) + internal_nonroot
    decisions = 2 * (n + internal_nonroot)
    return calls, decisions

def recursive_reference(adjacency):
    matrix = adjacency.tolist() if hasattr(adjacency, 'tolist') else adjacency
    def visit(row, lo, hi):
        if hi-lo == 1:
            return 1, 0
        mid = (lo+hi)//2
        calls, decisions = 1, 2
        for a, b in [(lo, mid), (mid, hi)]:
            if any(row[a:b]):
                c, d = visit(row, a, b)
                calls += c
                decisions += d
        return calls, decisions
    values = [visit(row, 0, len(row)) for row in matrix]
    return tuple(map(sum, zip(*values)))

def sample_graphs(batch, side, generator, device, family='community'):
    if side < 4 or side & (side-1):
        raise ValueError('Partition size must be a power of two >= 4')
    blocks, expected_degree = 4, 4.
    mixing = .1 + .2*torch.rand(batch, 1, 1, generator=generator, device=device)
    if family == 'er':
        mixing.fill_(.75)
    elif family != 'community':
        raise ValueError(family)
    group = torch.arange(side, device=device)//(side//blocks)
    same = group[:, None] == group[None, :]
    p_in = (1-mixing)*expected_degree/(side//blocks)
    p_out = mixing*expected_degree/(side-side//blocks)
    p = torch.where(same, p_in, p_out)
    b = torch.rand(batch, side, side, generator=generator, device=device) < p
    # Labels, unlike community memberships, are independent random coordinates.
    ru = torch.rand(batch, side, generator=generator, device=device).argsort(dim=-1)
    rv = torch.rand(batch, side, generator=generator, device=device).argsort(dim=-1)
    b = b.gather(1, ru[:, :, None].expand(-1, -1, side))
    b = b.gather(2, rv[:, None, :].expand(-1, side, -1))
    a = torch.zeros(batch, 2*side, 2*side, dtype=torch.bool, device=device)
    a[:, :side, side:] = b
    a[:, side:, :side] = b.transpose(1, 2)
    return a

def random_orders(batch, side, generator, device):
    u = torch.rand(batch, side, generator=generator, device=device).argsort(dim=-1)
    v = torch.rand(batch, side, generator=generator, device=device).argsort(dim=-1)+side
    return torch.cat([u, v], dim=-1)

def reorder(adjacency, permutation):
    n = permutation.shape[-1]
    return adjacency.gather(1, permutation[:, :, None].expand(-1, -1, n)).gather(
        2, permutation[:, None, :].expand(-1, n, -1))

def swap_permutations(permutation, pairs):
    out = permutation.clone()
    batch = torch.arange(len(out), device=out.device)
    out[batch, pairs[:, 0]] = permutation[batch, pairs[:, 1]]
    out[batch, pairs[:, 1]] = permutation[batch, pairs[:, 0]]
    return out

def legal_pairs(side, device):
    one = torch.triu_indices(side, side, offset=1, device=device).T
    return torch.cat([one, one+side], dim=0)
