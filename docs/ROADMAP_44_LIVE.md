# Roadmap 44: selected live verified RGB mirror

Requires a Roadmap 61 model-selection JSON, the **exact** checkpoint SHA from selection and a real camera. Run:
`python -m experiments.roadmap_44_live_mirror --selection runs/roadmap-61/selection.json --checkpoint path/to/trained.npz --camera 0 --lead 10 --fps 10 --no-display`.

At every acquired frame: resolve forecasts issued at earlier frame indices, append the actual frame, predict from the observed prefix, materialize model + persistence + image-velocity forecasts, timestamp **after inference**, and commit to pending queue. Report real measured elapsed milliseconds from commit to target and JAX/CPU inference time separately. Dropped frames change elapsed time; a 10-frame lead is not necessarily one second. Never select/tune using the live test session. CI uses a deterministic offline harness, not a claim of an actual webcam run. The optional display shows issued forecast versus actual and error; it does not overlay future camera truth before the target arrives.
