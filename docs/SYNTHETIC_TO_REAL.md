# Synthetic-to-real visual curriculum: a transfer test, not a result

Our original synthetic sheet is an eight-channel 16x16 model; the real webcam
predictor is a different 16-channel 32x32 model. **No weights were transferred.**
This new experiment instead trains the *same* video-residual architecture first
on procedurally generated 2D video and then on a real recording.

Four deterministic stages: one moving disc; two discs; an occluding wall;
a moving camera. The network receives **rendered frames only** (not dot coordinates,
visibility masks, object identities, world geometry, or simulator seeds).
The camera shifts at stage 3 but the current video model has no known-action
input from the real camera.

Run locally; recordings remain ignored by git:

```bash
python -m experiments.synthetic_to_video --input recordings/webcam.npz \
  --pretrain-steps 100 --finetune-steps 100 --out runs/synthetic-transfer
cat runs/synthetic-transfer/metrics.json
```

The two arms share initial weights, architecture, fine-tuning batches/seed,
training objective and fine-tuning compute. The pretrained arm additionally
receives 100 synthetic updates. Thus a positive gain would be *consistent*
with useful transfer but might reflect **additional compute**; follow up with
a 200-real-step compute-matched scratch control and independent recording.
Report whole-frame and motion-region errors plus persistence, and never claim
3D geometry or object permanence from these pixel metrics alone.
