# One-second persistence during occlusion and interactions

Frozen-checkpoint stress test: compare the same model and observation-only
baselines on unseen clear, stripe-occluded and two-ball sequences.

Run: `python -m experiments.rgb_occlusion_probe --checkpoint runs/rgb-curriculum/seed-11-curriculum.npz`

The world now marks a ball as visible only when some pixels survive the
foreground occlusion stripe. This visibility label is **never a model input**.
Report how many episodes truly contain reappearance; if zero, do not
claim the test measured reappearance tracking.

Gate: better reappearance prediction and motion-region error than
persistence and constant velocity on independent, event-rich scenes.
No new NCA architecture is introduced.
