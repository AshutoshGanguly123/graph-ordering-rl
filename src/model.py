"""Small position-aware graph encoder and symmetric legal-swap scorer."""
import torch
from torch import nn
from environment import legal_pairs

class SwapPolicy(nn.Module):
    def __init__(self, side=64, width=64, layers=3, head_width=128):
        super().__init__()
        self.side, self.width = side, width
        self.register_buffer('pairs', legal_pairs(side, 'cpu'))
        index = torch.arange(2*side)
        features = torch.stack([(index < side).float(), (index >= side).float(),
                                (index % side).float()/(side-1)], dim=-1)
        self.register_buffer('static_features', features)
        self.input = nn.Linear(4, width)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(2*width, width), nn.ReLU(),
                                                 nn.Linear(width, width)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(3*width+3, head_width), nn.ReLU(), nn.Linear(head_width, 1))

    def forward(self, adjacency, progress, best_fraction, return_embeddings=False):
        a = adjacency.float()
        degree = a.sum(dim=-1, keepdim=True)
        inputs = torch.cat([self.static_features[None].expand(len(a), -1, -1),
                            degree.log1p()], dim=-1)
        h = self.input(inputs).relu()
        norm_a = a / degree.clamp_min(1)
        for layer, norm in zip(self.layers, self.norms):
            message = torch.bmm(norm_a.to(h.dtype), h)
            h = norm(h + layer(torch.cat([h, message], dim=-1))).relu()
        first, second = self.pairs[:, 0], self.pairs[:, 1]
        left, right = h[:, first], h[:, second]
        distance = (first-second).abs().float()/(self.side-1)
        extra = torch.stack([distance[None].expand(len(a), -1),
                             progress[:, None].expand(-1, len(first)),
                             best_fraction[:, None].expand(-1, len(first))], dim=-1).to(h.dtype)
        pair_features = torch.cat([left+right, (left-right).abs(), left*right, extra], dim=-1)
        logits = self.head(pair_features).squeeze(-1).float()
        return (logits, h) if return_embeddings else logits
