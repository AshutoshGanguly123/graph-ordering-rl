# E007 learned-teacher cycle

Read PRD.md, config.json, rl_config.json and HANDOFF.md. User authorized the complete
learned-teacher -> fresh same-architecture student SFT -> matched RL experiment.
No agents, scheduler, progress monitor, STOP or critic. Preserve all older runs.
Immutable scientific files after deployment; source changes require a new identity.
Fresh graphs and held-out splits; never train on previous test graphs. Teacher
is E006 selected RL update 950, not the supervised 9,500 checkpoint. Always verify
its SHA256. Four physically empty unreserved Ada GPUs only, direct named tmux.
Never print/upload credentials or the private HPC helper/runbook.
