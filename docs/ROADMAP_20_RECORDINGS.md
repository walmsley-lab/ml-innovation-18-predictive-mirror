# Roadmap 20 — independent camera recording replay

Record two **different acquisition sessions**, for example on different days, and export each as `np.savez(..., frames=rgb_float32_T_H_W_3, timestamps_ns=monotonic_int64_T)`. Preserve camera identity, crop/resizing rule, lighting, calibration, capture gaps and raw immutable file hash. The loader refuses absent/nonmonotonic timestamps and identical file content.

`python -m experiments.roadmap_20_recording_eval --calibration first.npz --test second.npz --model nca --checkpoint trained.npz`

Forecasts receive only frames before the target, then compare to the later recorded frame; report real captured lead milliseconds, not nominal 10 fps. This is post-hoc **offline replay**, not evidence of real-time inference or within-deadline commitment. `--smoke` explicitly labels untrained inference as functional diagnostics. Separate acquisition sessions require operator verification beyond unequal hashes; do not tune on the held-out session.
