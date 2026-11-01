# Curriculum training for +10 frame RGB motion

Paired arms, **same** neural cellular automaton, initial weights, optimizer,
training updates and held-out identities. The only experimental variable is
training diversity: one repeated trajectory versus fresh seeded episodes
progressing through single object, occlusion, camera motion and two objects.

Run: `python -m experiments.rgb_curriculum --steps 100 --horizon 10`

Both arms select checkpoint on disjoint validation scenes, including step zero.
The final test set uses distinct scenes (71000+). Do not treat the result as
better than constant velocity until its multi-seed +10 test error demonstrates
that, including moving pixels. Freeze the benchmark and report negative runs.
This runner has no promise of improved learning or CPU throughput.
