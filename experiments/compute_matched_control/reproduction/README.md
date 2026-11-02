# Small reproduction pack

Contains pointers to immutable configs and the E007 audited eval artifacts already in `../../review/` and `../../metrics/`.

```bash
python ../../review/analyze.py   # regenerates comparison tables from audited JSON when evidence present
python ../prepare_configs.py --seed 202611020 --arm teacher_continued_rl
```
