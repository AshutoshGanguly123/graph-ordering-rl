# Reference research papers

Open-access / arXiv e-prints that frame this repo’s method, evaluation, and the broader Transformer-over-graphs program.

## Graph Transformers — capabilities / incapacities / ordering

| Paper | Venue | Why it is here | File / link |
|-------|-------|----------------|-------------|
| Order Matters: Sequence to Sequence for Sets (Vinyals et al.) | ICLR 2016 | Order sensitivity of seq models on sets — background for *learned graph orderings* | [arxiv.org/abs/1511.06391](https://arxiv.org/abs/1511.06391) · `order_matters_1511.06391.pdf` |
| Pure Transformers are Powerful Graph Learners (Kim et al.) | NeurIPS 2022 | Pure Transformer tokenization as a graph learner | [arxiv.org/abs/2207.02505](https://arxiv.org/abs/2207.02505) · `pure_transformers_graph_2207.02505.pdf` |
| Recipe for a General, Powerful, Scalable Graph Transformer / **GraphGPS** (Rampášek et al.) | NeurIPS 2022 | Hybrid MPNN + attention GPS architecture | [arxiv.org/abs/2205.12454](https://arxiv.org/abs/2205.12454) · `graphgps_2205.12454.pdf` |
| **GDGB**: A Benchmark for Generative Dynamic Text-Attributed Graph Learning | arXiv 2025 | Generative / dynamic graph stress tests beyond static setups | [arxiv.org/abs/2507.03267](https://arxiv.org/abs/2507.03267) · `gdgb_2507.03267.pdf` |

## RL / policy optimization (this experiment stack)

| Paper | Venue | Why | File / link |
|-------|-------|-----|-------------|
| Proximal Policy Optimization Algorithms (Schulman et al.) | arXiv 2017 | Clipped PG baseline this stack contrasts with (critic-free RLOO) | `ppo_schulman_1707.06347.pdf` |
| Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback (Ahmadian et al.) | ACL 2024 | RLOO / REINFORCE-style objectives | `rloo_ahmadian_2402.14740.pdf` |
| Soft Actor-Critic (Haarnoja et al.) | ICML 2018 | Max-ent RL background | `sac_haarnoja_1801.01290.pdf` |
| Attention, Learn to Solve Routing Problems! (Kool et al.) | ICLR 2019 | Learned constructive policies on combinatorial graphs | `kool_routing_1803.08475.pdf` |
| Pointer Networks (Vinyals et al.) | NeurIPS 2015 | Discrete combinatorial sequence outputs | `pointer_networks_1506.03134.pdf` |
| Graph Attention Networks (Veličković et al.) | ICLR 2018 | Structure-aware attention | `gat_1710.10903.pdf` |

PDFs are arXiv e-prints for offline reading. Prefer abstract pages for citations.
