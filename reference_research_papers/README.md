# Reference research papers

Open-access / arXiv e-prints that frame this repo’s method and evaluation.

| Paper | Venue | Why it is here | File / link |
|-------|-------|----------------|-------------|
| Proximal Policy Optimization Algorithms (Schulman et al.) | arXiv 2017 (widely adopted RL baseline) | Canonical clipped policy-gradient baseline this stack deliberately contrasts with (we use critic-free RLOO) | [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) · `ppo_schulman_1707.06347.pdf` |
| Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback (Ahmadian et al.) | ACL 2024 | RLOO / REINFORCE-style objectives without a learned critic — closest algorithmic family to our dense RLOO updates | [arxiv.org/abs/2402.14740](https://arxiv.org/abs/2402.14740) · `rloo_ahmadian_2402.14740.pdf` |
| Soft Actor-Critic (Haarnoja et al.) | ICML 2018 | Maximum-entropy RL context for continuous / stochastic policies used as background for modern policy optimization | [arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290) · `sac_haarnoja_1801.01290.pdf` |
| Attention, Learn to Solve Routing Problems! (Kool et al.) | ICLR 2019 | Learned constructive policies for combinatorial graph problems — lineage for graph-structured RL | [arxiv.org/abs/1803.08475](https://arxiv.org/abs/1803.08475) · `kool_routing_1803.08475.pdf` |
| Pointer Networks (Vinyals et al.) | NeurIPS 2015 | Sequence models over discrete combinatorial outputs | [arxiv.org/abs/1506.03134](https://arxiv.org/abs/1506.03134) · `pointer_networks_1506.03134.pdf` |
| Graph Attention Networks (Veličković et al.) | ICLR 2018 | Standard graph encoder building block for structure-aware policies | [arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903) · `gat_1710.10903.pdf` |

PDFs are arXiv e-prints for offline reading. Prefer the abstract pages for citations.
