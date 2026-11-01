# Independent RGB webcam transfer

This is an evaluation-only CLI for a **frozen** trained checkpoint.
Use a new RGB recording not used for training/validation. Explicitly
list known training recording paths in `--exclude-training`; it compares
file hashes and rejects exact reuse. The check cannot detect *all*
semantic overlap or leakage, so record dataset provenance.

Example:
```bash
python -m experiments.rgb_transfer_eval \
  --checkpoint runs/rgb-curriculum/seed-11-curriculum.npz \
  --recordings recordings/rgb-evaluation-1.npz \
  --exclude-training recordings/rgb-webcam.npz \
  --horizon 10
```

Uses final 20% of *each* recording, context and future fully inside it.
Reports +1/+3/+5/+10 RGB and motion-region MSE versus persistence and
image-derived constant velocity, and measured median capture interval.

Use independent captures, deliberate hand/ball motions and multiple seeds.
A 1-second claim requires measured timestamps, not the requested FPS alone.
