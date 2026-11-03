# PR 56 — Developmental ordering, without extra training exposure

A separate RGB NCA experiment asks whether *when* a model sees different
kinds of motion changes what it learns.

Three synthetic motion families are defined in the existing RGB world:
single-object simple motion; one ball with occlusion; and two-ball
interaction with occlusion and camera movement.

**Staged** and **shuffled** arms receive exactly the same unique batches
of pixel observations, identical initial parameters, optimizer and number
of updates. The only experimental variable is the order of their
presentations. The validation score is averaged across all three scenario
families, and checkpoint step zero remains eligible in both arms.
Final tests use separate seed ranges and per-scenario +1/+3/+5/+10
metrics against observation-only persistence and constant velocity.

    python -m experiments.rgb_developmental_curriculum \
      --steps 30 --context 8 --horizon 10 \
      --out runs/rgb-developmental-curriculum

    python -m unittest tests.test_rgb_developmental_curriculum -v

Outputs include separate checkpoints, an RGB sample per arm,
scenario-specific metrics and the exact training stage order for every
seed. A win on this synthetic curriculum does not establish that
developmental progression is biologically necessary; it merely motivates
a more constrained follow-up with other worlds or independent videos.
